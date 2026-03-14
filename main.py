import os
import json
import logging
import threading
import random
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

# メイン返信ロジック
def process_and_push_reply(user_id: str, user_text: str, motif_label: Optional[str] = None, selected_date: Optional[str] = None, selected_time: Optional[str] = None) -> None:
    try:
        user_ref = db.collection("users").document(user_id)
        user_data = user_ref.get().to_dict() or {}
        user_name = user_data.get("name")

        # 1. リセット
        if user_text == "リセット":
            user_ref.delete()
            line_bot_api.push_message(user_id, TextSendMessage(text="ようこそ、探究者の方。新たな観測を始めましょう。\nまずは、あなた様をどのようにお呼びすればよろしいですか？（本名でもニックネームでも構いません）"))
            return

        # 2. 名前登録
        if not user_name:
            user_ref.set({"name": user_text}, merge=True)
            send_birthday_picker(user_id, f"……{user_text}様ですね。心に刻みました。次に、あなたの生まれた日を教えてください。")
            return

        # 3. 生年月日・時刻登録
        if not user_data.get("birth_date"):
            if selected_date:
                user_ref.set({"birth_date": selected_date}, merge=True)
                send_time_picker(user_id)
            else: send_birthday_picker(user_id, f"{user_name}様。観測を始める前に、生まれた日を教えてください。")
            return

        if user_data.get("birth_hour") is None:
            if selected_time: 
                user_ref.set({"birth_hour": int(selected_time.split(":")[0])}, merge=True)
                line_bot_api.push_message(user_id, TextSendMessage(text=f"刻印が完成しました。今、{user_name}様が一番視たいことは何でしょうか。"))
            elif user_text == "UNKNOWN_TIME": 
                user_ref.set({"birth_hour": 12}, merge=True)
                line_bot_api.push_message(user_id, TextSendMessage(text=f"承知いたしました。では、今{user_name}様が視たいことは何でしょうか。"))
            else: send_time_picker(user_id)
            return

        # 4. 深掘り・モチーフ提示フェーズ
        if not user_data.get("is_dialogue_mode") and motif_label is None:
            if user_data.get("temp_category"):
                # 深掘りへの回答を受け取った
                combined_consult = f"{user_data['temp_category']}（詳細：{user_text}）"
                user_ref.update({"pending_consult": combined_consult, "temp_category": firestore.DELETE_FIELD})
            elif not user_data.get("pending_consult"):
                # 新しい相談
                if len(user_text) <= 5:
                    user_ref.update({"temp_category": user_text})
                    line_bot_api.push_message(user_id, TextSendMessage(text=f"……「{user_text}」についてですね。その奥にある想いを、もう少しだけ詳しく教えていただけますか？（例：具体的な状況や、今感じていることなど）"))
                    return
                else:
                    user_ref.update({"pending_consult": user_text})

            # モチーフをランダムに4つ提示
            sampled = random.sample(ALL_MOTIFS, 4)
            buttons = [QuickReplyButton(action=PostbackAction(label=m, data=f"action=select_motif&label={m}", display_text=m)) for m in sampled]
            line_bot_api.push_message(user_id, TextSendMessage(text="準備は整いました。心に触れる象徴を一つ選んでください。", quick_reply=QuickReply(items=buttons)))
            return

        # 5. 鑑定実行（モチーフ選択直後）
        if motif_label:
            profile = {"name": user_name, "birth_year": int(user_data["birth_date"].split("-")[0]), "birth_month": int(user_data["birth_date"].split("-")[1]), "birth_day": int(user_data["birth_date"].split("-")[2]), "birth_hour": user_data["birth_hour"]}
            consult_text = user_data.get("pending_consult", "これからの運勢")
            
            result = oracle_engine.predict(profile, consult_text, motif_label, is_dialogue=False)
            
            user_ref.update({"is_dialogue_mode": True, "chat_history": f"識の神託: {result['message']}\n", "last_motif": motif_label, "pending_consult": firestore.DELETE_FIELD})
            line_bot_api.push_message(user_id, TextSendMessage(text=result["message"]))
            return

        # 6. 対話・カウンセリングモード
        if user_data.get("is_dialogue_mode"):
            history = user_data.get("chat_history", "")
            profile = {"name": user_name, "birth_year": int(user_data["birth_date"].split("-")[0])}
            
            result = oracle_engine.predict(profile, user_text, user_data.get("last_motif"), is_dialogue=True, chat_history=history)
            reply_text = result["message"]

            # AIが「終了」と判断したかのフラグチェック
            if "[END_SESSION]" in reply_text:
                reply_text = reply_text.replace("[END_SESSION]", "").strip()
                user_ref.update({"is_dialogue_mode": False, "chat_history": ""})
                line_bot_api.push_message(user_id, TextSendMessage(text=reply_text))
            else:
                user_ref.update({"chat_history": history + f"{user_name}: {user_text}\n識: {reply_text}\n"})
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
    if query.get("action") == "select_motif":
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
