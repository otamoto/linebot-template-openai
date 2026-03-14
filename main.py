import os
import json
import logging
import threading
import random
import re
import unicodedata
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from linebot import LineBotApi, WebhookHandler
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, 
    QuickReply, QuickReplyButton, PostbackAction, PostbackEvent,
    DatetimePickerAction, TemplateSendMessage, ButtonsTemplate
)
from linebot.exceptions import InvalidSignatureError
from google import genai
import firebase_admin
from firebase_admin import credentials, firestore
from oracle_engine import OracleEngine

# 初期化
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s : %(message)s")
logger = logging.getLogger(__name__)
app = FastAPI()

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FIREBASE_SERVICE_ACCOUNT_JSON = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-1.5-flash")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
genai_client = genai.Client(api_key=GEMINI_API_KEY)

if not firebase_admin._apps:
    key_dict = json.loads(FIREBASE_SERVICE_ACCOUNT_JSON)
    firebase_admin.initialize_app(credentials.Certificate(key_dict))
db = firestore.client()

oracle_engine = OracleEngine(gemini_client=genai_client, model_name=CHAT_MODEL)

# 40種のモチーフ
ALL_MOTIFS = [
    "銀の鍵", "砂時計", "古びた鏡", "聖なる滴", "琥珀の蝶", "折れた剣", "青い月", "羅針盤", "封じられた手紙", "金の天秤",
    "揺れる灯火", "水晶の髑髏", "沈黙の鐘", "茨の冠", "無垢な羽根", "双子の蛇", "星の砂", "錆びた歯車", "輝く聖杯", "黒い薔薇",
    "白い孔雀", "秘められた林檎", "古の地図", "時の歯車", "深海の真珠", "暁の鳥", "黄昏の雫", "不滅の炎", "氷の心臓", "奏でる竪琴",
    "空の器", "叡智の梟", "秘密の鍵穴", "約束の指輪", "流れる雲", "不動の岩", "踊る影", "導きの杖", "遮られた眼", "語らぬ仮面"
]

def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", (text or "").strip())

def send_birthday_picker(user_id: str, message: str):
    date_picker = ButtonsTemplate(text=message, actions=[DatetimePickerAction(label="カレンダーで選択", data="action=set_birthday", mode="date", initial="1995-01-01")])
    line_bot_api.push_message(user_id, TemplateSendMessage(alt_text="誕生日選択", template=date_picker))

def send_time_picker(user_id: str):
    time_picker = ButtonsTemplate(text="生まれた時刻は分かりますか？", actions=[DatetimePickerAction(label="時刻を選択", data="action=set_birthtime", mode="time", initial="12:00"), PostbackAction(label="分からない", data="action=set_birthtime_unknown")])
    line_bot_api.push_message(user_id, TemplateSendMessage(alt_text="時刻選択", template=time_picker))

def send_profile_confirm(user_id: str, date_str: str, time_str: str):
    text = f"生年月日: {date_str}\n出生時刻: {time_str}\n\nこちらの刻印でよろしいでしょうか？"
    buttons = [
        QuickReplyButton(action=PostbackAction(label="はい", data="action=confirm_profile&res=yes", display_text="はい")),
        QuickReplyButton(action=PostbackAction(label="やり直す", data="action=confirm_profile&res=no", display_text="やり直す"))
    ]
    line_bot_api.push_message(user_id, TextSendMessage(text=text, quick_reply=QuickReply(items=buttons)))

def build_user_profile(user_data: dict) -> dict:
    profile = {"name": user_data.get("name", "あなた")}
    if "birth_date" in user_data:
        y, m, d = map(int, user_data["birth_date"].split("-"))
        profile.update({"birth_year": y, "birth_month": m, "birth_day": d})
    if "birth_hour" in user_data:
        profile["birth_hour"] = user_data["birth_hour"]
    return profile

# メイン返信ロジック
def process_and_push_reply(user_id: str, user_text: str, motif_label: Optional[str] = None, selected_date: Optional[str] = None, selected_time: Optional[str] = None) -> None:
    try:
        user_ref = db.collection("users").document(user_id)
        user_data = user_ref.get().to_dict() or {}
        user_name = user_data.get("name")

        # 1. リセット
        if user_text == "リセット":
            user_ref.delete()
            line_bot_api.push_message(user_id, TextSendMessage(text="ようこそ、探究者の方。新たな観測を始めましょう。\nまずは、あなた様をどのようにお呼びすればよろしいですか？（『〇〇です』などは付けず、お呼びするお名前のみを送信してください）"))
            return

        # 2. 名前登録
        if not user_name:
            clean_name = re.sub(r"(です|と申します|だよ|と申す|だ|といいます|って呼びます|って呼んで)$", "", user_text).strip()
            user_ref.set({"name": clean_name}, merge=True)
            send_birthday_picker(user_id, f"……{clean_name}様ですね。心に刻みました。次に、あなたの生まれた日を教えてください。")
            return

        # 3. 生年月日・時刻の登録と確認フロー
        if not user_data.get("is_profile_confirmed"):
            if not user_data.get("birth_date"):
                if selected_date:
                    user_ref.set({"birth_date": selected_date}, merge=True)
                    send_time_picker(user_id)
                else: 
                    send_birthday_picker(user_id, f"{user_name}様。観測を始める前に、生まれた日を教えてください。")
                return

            if user_data.get("birth_hour") is None:
                if selected_time: 
                    h = int(selected_time.split(":")[0])
                    user_ref.set({"birth_hour": h}, merge=True)
                    send_profile_confirm(user_id, user_data.get("birth_date"), f"{h}時頃")
                elif user_text == "UNKNOWN_TIME": 
                    user_ref.set({"birth_hour": 12}, merge=True)
                    send_profile_confirm(user_id, user_data.get("birth_date"), "不明（正午として計算）")
                else: 
                    send_time_picker(user_id)
                return

            if user_text == "CONFIRM_YES":
                user_ref.set({"is_profile_confirmed": True}, merge=True)
                line_bot_api.push_message(user_id, TextSendMessage(text=f"刻印が完成しました。今、{user_name}様が一番視たいことは何でしょうか。"))
                return
            elif user_text == "CONFIRM_NO":
                user_ref.set({"birth_date": None, "birth_hour": None}, merge=True)
                send_birthday_picker(user_id, "承知いたしました。では、もう一度生まれた日を正しく教えてください。")
                return
            else:
                send_profile_confirm(user_id, user_data.get("birth_date"), f"{user_data.get('birth_hour')}時頃" if user_data.get("birth_hour") != 12 else "不明（正午として計算）")
                return

        # 4. 「新たなる観測」確認 ＆ 深掘りフェーズ
        if not user_data.get("is_dialogue_mode") and motif_label is None:
            
            # 再開ボタン（はい/いいえ）の処理
            if user_text == "RESTART_YES":
                user_text = user_data.get("temp_restart_text", "これからの運勢")
                user_ref.set({"temp_restart_text": None}, merge=True)
            elif user_text == "RESTART_NO":
                user_ref.set({"temp_restart_text": None}, merge=True)
                line_bot_api.push_message(user_id, TextSendMessage(text="承知いたしました。私はまた淵にてお待ちしております。"))
                return
            elif not user_data.get("pending_consult") and not user_data.get("temp_category"):
                # 通常の入力が来た場合、まずは「始めるか」を確認する
                user_ref.set({"temp_restart_text": user_text}, merge=True)
                buttons = [
                    QuickReplyButton(action=PostbackAction(label="はい", data="action=restart&res=yes", display_text="はい")),
                    QuickReplyButton(action=PostbackAction(label="いいえ", data="action=restart&res=no", display_text="いいえ"))
                ]
                line_bot_api.push_message(user_id, TextSendMessage(text="新たなる観測を始めますか？", quick_reply=QuickReply(items=buttons)))
                return

            # 「はい」が押された後の深掘り処理
            if user_data.get("temp_category"):
                combined_consult = f"{user_data['temp_category']}（詳細：{user_text}）"
                user_ref.set({"pending_consult": combined_consult, "temp_category": None}, merge=True)
            elif not user_data.get("pending_consult"):
                if len(user_text) <= 15:
                    user_ref.set({"temp_category": user_text}, merge=True)
                    line_bot_api.push_message(user_id, TextSendMessage(text=f"……「{user_text}」についてですね。その奥にある想いを、もう少しだけ詳しく教えていただけますか？\n（具体的な状況や、今感じている不安などを教えていただけると、より深く観測できます）"))
                    return
                else:
                    user_ref.set({"pending_consult": user_text}, merge=True)

            # モチーフの提示
            sampled = random.sample(ALL_MOTIFS, 4)
            buttons = [QuickReplyButton(action=PostbackAction(label=m, data=f"action=select_motif&label={m}", display_text=m)) for m in sampled]
            line_bot_api.push_message(user_id, TextSendMessage(text="準備は整いました。心に触れる象徴を一つ選んでください。", quick_reply=QuickReply(items=buttons)))
            return

        # 5. 鑑定実行（モチーフ選択直後）
        if motif_label:
            profile = build_user_profile(user_data)
            consult_text = user_data.get("pending_consult", "これからの運勢")
            
            result = oracle_engine.predict(profile, consult_text, motif_label, is_dialogue=False)
            
            user_ref.set({"is_dialogue_mode": True, "chat_history": f"識の神託: {result['message']}\n", "last_motif": motif_label, "pending_consult": None}, merge=True)
            line_bot_api.push_message(user_id, TextSendMessage(text=result["message"]))
            return

        # 6. 対話・カウンセリングモード
        if user_data.get("is_dialogue_mode"):
            history = user_data.get("chat_history", "")
            profile = build_user_profile(user_data)
            
            result = oracle_engine.predict(profile, user_text, user_data.get("last_motif", "静かなる光"), is_dialogue=True, chat_history=history)
            reply_text = result["message"]

            if "[END_SESSION]" in reply_text:
                reply_text = reply_text.replace("[END_SESSION]", "").strip()
                # 終了時は確実に初期化（Noneをセットしてエラーを防ぐ）
                user_ref.set({"is_dialogue_mode": False, "chat_history": None, "temp_category": None, "pending_consult": None, "temp_restart_text": None}, merge=True)
                line_bot_api.push_message(user_id, TextSendMessage(text=reply_text))
            else:
                user_ref.set({"chat_history": history + f"{user_name}: {user_text}\n識: {reply_text}\n"}, merge=True)
                line_bot_api.push_message(user_id, TextSendMessage(text=reply_text))
            return

    except Exception as e:
        logger.exception("Error")
        line_bot_api.push_message(user_id, TextSendMessage(text="識の視界が揺らぎました。"))

# -------------------------
# コールバック処理
# -------------------------
@app.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature")
    body = await request.body()
    try:
        handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400)
    return JSONResponse({"ok": True})

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event: MessageEvent):
    threading.Thread(target=process_and_push_reply, args=(event.source.user_id, normalize_text(event.message.text))).start()

@handler.add(PostbackEvent)
def handle_postback(event: PostbackEvent):
    user_id = event.source.user_id
    query = dict(x.split('=') for x in event.postback.data.split('&'))
    
    if query.get("action") == "confirm_profile":
        text_val = "CONFIRM_YES" if query.get("res") == "yes" else "CONFIRM_NO"
        threading.Thread(target=process_and_push_reply, args=(user_id, text_val)).start()
    elif query.get("action") == "restart":
        text_val = "RESTART_YES" if query.get("res") == "yes" else "RESTART_NO"
        threading.Thread(target=process_and_push_reply, args=(user_id, text_val)).start()
    elif query.get("action") == "select_motif":
        threading.Thread(target=process_and_push_reply, args=(user_id, "", query.get("label"))).start()
    elif query.get("action") == "set_birthday":
        threading.Thread(target=process_and_push_reply, args=(user_id, "", None, event.postback.params.get("date"))).start()
    elif query.get("action") == "set_birthtime":
        threading.Thread(target=process_and_push_reply, args=(user_id, "", None, None, event.postback.params.get("time"))).start()
    elif query.get("action") == "set_birthtime_unknown":
        threading.Thread(target=process_and_push_reply, args=(user_id, "UNKNOWN_TIME")).start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
