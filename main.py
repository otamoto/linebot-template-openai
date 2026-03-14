import os
import json
import logging
import unicodedata
import threading
import re
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any

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

# -------------------------
# 初期化
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s : %(message)s")
logger = logging.getLogger(__name__)
app = FastAPI(title="SHIKI LINE Bot")

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
    cred = credentials.Certificate(key_dict)
    firebase_admin.initialize_app(cred)
db = firestore.client()

oracle_engine = OracleEngine(gemini_client=genai_client, model_name=CHAT_MODEL)

MOTIFS = {"silver_key": "銀の鍵", "hourglass": "砂時計", "ancient_mirror": "古びた鏡", "holy_drop": "聖なる滴"}

# -------------------------
# ユーティリティ & UI
# -------------------------
def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", (text or "").strip())

def build_user_profile(user_data: dict) -> dict:
    bd = user_data.get("birth_date", "1990-01-01")
    y, m, d = map(int, bd.split("-"))
    profile = {"birth_year": y, "birth_month": m, "birth_day": d}
    if user_data.get("birth_hour") is not None:
        profile["birth_hour"] = int(user_data["birth_hour"])
    return profile

def send_birthday_picker(user_id: str, message: str):
    date_picker = ButtonsTemplate(
        text=message,
        actions=[DatetimePickerAction(label="カレンダーで選択", data="action=set_birthday", mode="date", initial="1995-01-01")]
    )
    line_bot_api.push_message(user_id, TemplateSendMessage(alt_text="生年月日を選択", template=date_picker))

def send_time_picker(user_id: str):
    time_picker = ButtonsTemplate(
        text="生まれた「時刻」も分かりますか？より詳細な観測が可能になります。",
        actions=[
            DatetimePickerAction(label="時刻を選択する", data="action=set_birthtime", mode="time", initial="12:00"),
            PostbackAction(label="分からない", data="action=set_birthtime_unknown")
        ]
    )
    line_bot_api.push_message(user_id, TemplateSendMessage(alt_text="出生時間を選択", template=time_picker))

# -------------------------
# メイン返信ロジック
# -------------------------
def process_and_push_reply(user_id: str, user_text: str, motif_id: Optional[str] = None, selected_date: Optional[str] = None, selected_time: Optional[str] = None) -> None:
    try:
        user_ref = db.collection("users").document(user_id)
        user_data = user_ref.get().to_dict() or {}

        # 1. リセット
        if user_text == "リセット":
            user_ref.delete()
            send_birthday_picker(user_id, "すべての記録を虚空へ返しました。新たな観測を始めましょう。あなたの生まれた日はいつですか？")
            return

        # 2. 登録フロー
        if not user_data.get("birth_date"):
            if selected_date:
                user_ref.set({"birth_date": selected_date}, merge=True)
                send_time_picker(user_id)
            else:
                send_birthday_picker(user_id, "ようこそ。観測を始める前に、あなたの生まれた日を教えてください。")
            return

        if user_data.get("birth_hour") is None:
            if selected_time:
                h, m = map(int, selected_time.split(":"))
                user_ref.set({"birth_hour": h}, merge=True)
                line_bot_api.push_message(user_id, TextSendMessage(text="刻印が完成しました。今、あなたが一番視たいことを教えてください。"))
            elif user_text == "UNKNOWN_TIME":
                user_ref.set({"birth_hour": 12}, merge=True)
                line_bot_api.push_message(user_id, TextSendMessage(text="承知いたしました。では、今あなたが一番視たいことを教えてください。"))
            else:
                send_time_picker(user_id)
            return

        # 3. モチーフ選択 & 鑑定実行
        if motif_id:
            motif_label = MOTIFS.get(motif_id, "静かなる光")
            profile = build_user_profile(user_data)
            consult_text = user_data.get("pending_consult", "これからの運勢")
            
            result = oracle_engine.predict(profile, consult_text, motif_label, is_dialogue=False)
            
            # 初回鑑定を保存して対話モードへ
            user_ref.update({
                "is_dialogue_mode": True,
                "chat_history": f"識の神託: {result['message']}\n",
                "last_motif_label": motif_label,
                "dialogue_count": 1,
                "pending_consult": firestore.DELETE_FIELD
            })
            line_bot_api.push_message(user_id, TextSendMessage(text=result["message"]))
            return

        # 4. 対話モード（2回目以降）
        if user_data.get("is_dialogue_mode"):
            profile = build_user_profile(user_data)
            history = user_data.get("chat_history", "")
            motif_label = user_data.get("last_motif_label", "静かなる光")
            count = user_data.get("dialogue_count", 1)

            result = oracle_engine.predict(profile, user_text, motif_label, is_dialogue=True, chat_history=history)
            
            new_history = history + f"あなた: {user_text}\n識: {result['message']}\n"
            
            # クローズ判定（キーワード or 往復回数）
            close_keywords = ["ありがとう", "助かりました", "わかりました", "さようなら", "バイバイ", "やってみる"]
            is_closing = any(k in user_text for k in close_keywords) or count >= 5
            
            if is_closing:
                user_ref.update({"is_dialogue_mode": False, "chat_history": "", "dialogue_count": 0})
            else:
                user_ref.update({"chat_history": new_history, "dialogue_count": count + 1})
            
            line_bot_api.push_message(user_id, TextSendMessage(text=result["message"]))
            return

        # 5. 通常相談受付（モチーフ提示）
        user_ref.set({"pending_consult": user_text}, merge=True)
        buttons = [QuickReplyButton(action=PostbackAction(label=label, data=f"action=select_motif&id={m_id}", display_text=label)) for m_id, label in MOTIFS.items()]
        line_bot_api.push_message(user_id, TextSendMessage(text="準備は整いました。心に触れる象徴を一つ選んでください。", quick_reply=QuickReply(items=buttons)))

    except Exception:
        logger.exception("Error")
        line_bot_api.push_message(user_id, TextSendMessage(text="識の視界が揺らぎました。少し間を置いてください。"))

# -------------------------
# サーバー & コールバック
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
def handle_text_message(event: MessageEvent):
    user_id = event.source.user_id
    user_text = normalize_text(event.message.text)
    if user_text:
        threading.Thread(target=process_and_push_reply, args=(user_id, user_text)).start()

@handler.add(PostbackEvent)
def handle_postback(event: PostbackEvent):
    user_id = event.source.user_id
    query = dict(x.split('=') for x in event.postback.data.split('&'))
    
    if query.get("action") == "select_motif":
        threading.Thread(target=process_and_push_reply, args=(user_id, "", query.get("id"))).start()
    elif query.get("action") == "set_birthday":
        threading.Thread(target=process_and_push_reply, args=(user_id, "", None, event.postback.params.get("date"))).start()
    elif query.get("action") == "set_birthtime":
        threading.Thread(target=process_and_push_reply, args=(user_id, "", None, None, event.postback.params.get("time"))).start()
    elif query.get("action") == "set_birthtime_unknown":
        threading.Thread(target=process_and_push_reply, args=(user_id, "UNKNOWN_TIME")).start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
