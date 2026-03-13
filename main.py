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
    QuickReply, QuickReplyButton, MessageAction, PostbackAction,
    DatetimePickerAction, TemplateSendMessage, ButtonsTemplate, PostbackEvent
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

# モチーフ定義
MOTIFS = {
    "silver_key": "銀の鍵",
    "hourglass": "砂時計",
    "ancient_mirror": "古びた鏡",
    "holy_drop": "聖なる滴"
}

# ユーティリティ
def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", (text or "").strip())

def build_user_profile(user_data: dict) -> dict:
    bd = user_data.get("birth_date", "1990-01-01")
    y, m, d = map(int, bd.split("-"))
    profile = {"birth_year": y, "birth_month": m, "birth_day": d}
    if user_data.get("birth_hour") is not None:
        profile["birth_hour"] = int(user_data["birth_hour"])
        profile["birth_minute"] = int(user_data.get("birth_minute", 0))
    return profile

# UI送信関数
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

# メイン返信ロジック
def process_and_push_reply(user_id: str, user_text: str, motif_id: Optional[str] = None, selected_date: Optional[str] = None, selected_time: Optional[str] = None) -> None:
    try:
        user_ref = db.collection("users").document(user_id)
        user_data = user_ref.get().to_dict() or {}

        if user_text == "リセット":
            user_ref.delete()
            send_birthday_picker(user_id, "すべての記録を虚空へ返しました。新たな観測を始めましょう。あなたの生まれた日はいつですか？")
            return

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
                user_ref.set({"birth_hour": h, "birth_minute": m}, merge=True)
                line_bot_api.push_message(user_id, TextSendMessage(text="刻印が完成しました。今、あなたが一番視たいことを教えてください。"))
            elif user_text == "UNKNOWN_TIME":
                user_ref.set({"birth_hour": 12, "birth_minute": 0}, merge=True)
                line_bot_api.push_message(user_id, TextSendMessage(text="承知いたしました。では、日時の重なりを中心に観測します。今、あなたが一番視たいことを教えてください。"))
            else:
                send_time_picker(user_id)
            return

        if not motif_id:
            user_ref.set({"pending_consult": user_text}, merge=True)
            buttons = [QuickReplyButton(action=PostbackAction(label=label, data=f"action=select_motif&id={m_id}", display_text=label)) for m_id, label in MOTIFS.items()]
            line_bot_api.push_message(user_id, TextSendMessage(text="準備は整いました。あなたの直感を重ねます。いま、心に触れる象徴を一つ選んでください。", quick_reply=QuickReply(items=buttons)))
            return

        # 鑑定実行
        profile = build_user_profile(user_data)
        consult_text = user_data.get("pending_consult", "これからの運勢")
        motif_label = MOTIFS.get(motif_id, "静かなる光") # IDをラベルに変換
        
        result = oracle_engine.predict(profile, {"stress": 0.5}, consult_text, motif_label)
        
        line_bot_api.push_message(user_id, TextSendMessage(text=result.get("message", "……識の声が途切れました。")))
        user_ref.update({"pending_consult": firestore.DELETE_FIELD})

    except Exception:
        logger.exception("process_and_push_reply error")
        line_bot_api.push_message(user_id, TextSendMessage(text="識の視界が揺らぎました。少し間を置いてください。"))

# -------------------------
# LINE Callback & Event Handler
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
        threading.Thread(target=process_and_push_reply, args=(user_id, user_text), daemon=True).start()

@handler.add(PostbackEvent)
def handle_postback(event: PostbackEvent):
    user_id = event.source.user_id
    query = dict(x.split('=') for x in event.postback.data.split('&'))
    
    if query.get("action") == "select_motif":
        threading.Thread(target=process_and_push_reply, args=(user_id, "", query.get("id")), daemon=True).start()
    elif query.get("action") == "set_birthday":
        threading.Thread(target=process_and_push_reply, args=(user_id, "", None, event.postback.params.get("date")), daemon=True).start()
    elif query.get("action") == "set_birthtime":
        threading.Thread(target=process_and_push_reply, args=(user_id, "", None, None, event.postback.params.get("time")), daemon=True).start()
    elif query.get("action") == "set_birthtime_unknown":
        threading.Thread(target=process_and_push_reply, args=(user_id, "UNKNOWN_TIME"), daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
