import os
import json
import logging
import re
import threading
import unicodedata
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Dict, Any, List

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from linebot import LineBotApi, WebhookHandler
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, PostbackEvent,
    QuickReply, QuickReplyButton, MessageAction, DatetimePickerAction, PostbackAction
)
from linebot.exceptions import InvalidSignatureError
from google import genai
import firebase_admin
from firebase_admin import credentials, firestore

# 自作エンジンの読み込み
from oracle_engine import OracleEngine

# -------------------------
# 初期化
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s : %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="SHIKI LINE Bot")

# 環境変数
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FIREBASE_SERVICE_ACCOUNT_JSON = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
genai_client = genai.Client(api_key=GEMINI_API_KEY)

if not firebase_admin._apps:
    key_dict = json.loads(FIREBASE_SERVICE_ACCOUNT_JSON)
    cred = credentials.Certificate(key_dict)
    firebase_admin.initialize_app(cred)
db = firestore.client()

# エンジン初期化
oracle_engine = OracleEngine(genai_client)

# -------------------------
# ユーティリティ
# -------------------------
def build_user_profile(user_data: dict) -> dict:
    birth_date = user_data.get("birth_date", "1990-01-01")
    y, m, d = map(int, birth_date.split("-"))
    return {"birth_year": y, "birth_month": m, "birth_day": d}

def build_base_context(user_data: dict) -> dict:
    return {"stress": 0.5, "urgency": 0.5}

def slots_to_context(base, slots):
    return {**base, **(slots or {})}

def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", (text or "").strip())

# -------------------------
# 接続の核心
# -------------------------
def build_reading_reply(user_data: dict, active_text: str, known_slots: dict) -> Tuple[str, dict, dict]:
    profile = build_user_profile(user_data)
    context = slots_to_context(build_base_context(user_data), known_slots)
    
    # 儀式（モチーフ）の取得
    motif_id = user_data.get("ritual_motif_id", "静かなる光")

    # エンジン呼び出し
    result = oracle_engine.predict(profile, context, active_text, motif_id)
    
    reply_text = result["message"]
    if not user_data.get("birth_date"):
        reply_text += "\n\n……生まれた日の刻印を預ければ、より深く観測できるでしょう。"
    
    return reply_text, result, context

# -------------------------
# 以下、以前の handle_message_flow 等のロジックを継続
# (文字数制限のため主要なフローのみ記載します。不足分は以前のコードを維持してください)
# -------------------------

@app.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature")
    body = await request.body()
    try:
        handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400)
    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event: MessageEvent):
    u_id = event.source.user_id
    user_text = event.message.text.strip()
    
    user_ref = db.collection("users").document(u_id)
    user_data = user_ref.get().to_dict() or {}

    # 簡易ルーティング: 相談が来たら占う
    reply, result, ctx = build_reading_reply(user_data, user_text, {})
    
    user_ref.set({
        "last_active": datetime.now(timezone.utc),
        "last_oracle_message": reply
    }, merge=True)
    
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
