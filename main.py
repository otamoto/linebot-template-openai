import os
import json
import logging
import unicodedata
import threading
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from linebot.exceptions import InvalidSignatureError
from google import genai
import firebase_admin
from firebase_admin import credentials, firestore
from oracle_engine import OracleEngine

# 初期化
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s : %(message)s")
logger = logging.getLogger(__name__)
app = FastAPI(title="SHIKI LINE Bot")

# 環境変数
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FIREBASE_SERVICE_ACCOUNT_JSON = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
GEMINI_MODEL = os.getenv("CHAT_MODEL", "gemini-1.5-flash") # あなたの環境変数名 CHAT_MODEL に合わせました

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
genai_client = genai.Client(api_key=GEMINI_API_KEY)

if not firebase_admin._apps:
    key_dict = json.loads(FIREBASE_SERVICE_ACCOUNT_JSON)
    cred = credentials.Certificate(key_dict)
    firebase_admin.initialize_app(cred)
db = firestore.client()

oracle_engine = OracleEngine(gemini_client=genai_client, include_approx_sukuyo=False, model_name=GEMINI_MODEL)

# ユーティリティ
def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", (text or "").strip())

def build_user_profile(user_data: dict) -> dict:
    bd = user_data.get("birth_date", "1990-01-01")
    y, m, d = map(int, bd.split("-"))
    profile = {"birth_year": y, "birth_month": m, "birth_day": d}
    for key in ["birth_hour", "birth_minute", "birth_second"]:
        if user_data.get(key) is not None: profile[key] = int(user_data[key])
    profile["birth_longitude"] = float(user_data.get("birth_longitude", 135.0))
    return profile

def build_reading_reply(user_data: dict, active_text: str, known_slots: Optional[dict]) -> Tuple[str, dict, dict]:
    profile = build_user_profile(user_data)
    context = {**{"stress": 0.5, "urgency": 0.5}, **(known_slots or {})}
    motif_id = user_data.get("ritual_motif_id", "静かなる光")
    result = oracle_engine.predict(profile, context, active_text, motif_id)
    reply = result.get("message", "……識の声は、いまはまだ遠い。")
    if not user_data.get("birth_date"): reply += "\n\n……生まれた日の刻印を預ければ、より深く観測できるだろう。"
    return reply, result, context

def process_and_push_reply(user_id: str, user_text: str) -> None:
    try:
        user_ref = db.collection("users").document(user_id)
        user_data = user_ref.get().to_dict() or {}
        reply, result, ctx = build_reading_reply(user_data, user_text, {})
        user_ref.set({"last_active": datetime.now(timezone.utc), "last_oracle_message": reply, "updated_at": datetime.now(timezone.utc)}, merge=True)
        line_bot_api.push_message(user_id, TextSendMessage(text=reply))
    except Exception:
        logger.exception("process_and_push_reply error")
        line_bot_api.push_message(user_id, TextSendMessage(text="……識の視界が揺らぎました。少し間を置いて、もう一度問いかけてください。"))

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
