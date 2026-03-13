import os
import json
import logging
import unicodedata
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from linebot import LineBotApi, WebhookHandler
from linebot.models import (
    MessageEvent,
    TextMessage,
    TextSendMessage,
)
from linebot.exceptions import InvalidSignatureError

from google import genai

import firebase_admin
from firebase_admin import credentials, firestore

from oracle_engine import OracleEngine

# -------------------------
# 初期化
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s : %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="SHIKI LINE Bot")

# -------------------------
# 環境変数
# -------------------------
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FIREBASE_SERVICE_ACCOUNT_JSON = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not LINE_CHANNEL_ACCESS_TOKEN:
    raise RuntimeError("LINE_CHANNEL_ACCESS_TOKEN が設定されていません。")
if not LINE_CHANNEL_SECRET:
    raise RuntimeError("LINE_CHANNEL_SECRET が設定されていません。")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY が設定されていません。")
if not FIREBASE_SERVICE_ACCOUNT_JSON:
    raise RuntimeError("FIREBASE_SERVICE_ACCOUNT_JSON が設定されていません。")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
genai_client = genai.Client(api_key=GEMINI_API_KEY)

# -------------------------
# Firebase 初期化
# -------------------------
if not firebase_admin._apps:
    try:
        key_dict = json.loads(FIREBASE_SERVICE_ACCOUNT_JSON)
        cred = credentials.Certificate(key_dict)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        logger.exception("Firebase 初期化失敗")
        raise RuntimeError(f"Firebase 初期化失敗: {e}")

db = firestore.client()

# -------------------------
# エンジン初期化
# -------------------------
oracle_engine = OracleEngine(
    genai_client=genai_client,
    include_approx_sukuyo=False,
    model_name=GEMINI_MODEL,
)

# -------------------------
# ユーティリティ
# -------------------------
def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", (text or "").strip())


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def build_user_profile(user_data: dict) -> dict:
    """
    Firestoreの user_data から OracleEngine 向けプロフィールを構築する。
    想定フィールド:
    - birth_date: "YYYY-MM-DD"
    - birth_hour: 0-23
    - birth_minute: 0-59
    - birth_second: 0-59
    - birth_longitude: 経度（例 139.6917）
    """
    birth_date = user_data.get("birth_date", "1990-01-01")

    try:
        y, m, d = map(int, birth_date.split("-"))
    except Exception:
        logger.warning("birth_date の形式が不正: %s", birth_date)
        y, m, d = 1990, 1, 1

    profile = {
        "birth_year": y,
        "birth_month": m,
        "birth_day": d,
    }

    birth_hour = safe_int(user_data.get("birth_hour"), None)
    birth_minute = safe_int(user_data.get("birth_minute"), 0)
    birth_second = safe_int(user_data.get("birth_second"), 0)
    birth_longitude = safe_float(user_data.get("birth_longitude"), 135.0)

    if birth_hour is not None:
        profile["birth_hour"] = birth_hour
    if birth_minute is not None:
        profile["birth_minute"] = birth_minute
    if birth_second is not None:
        profile["birth_second"] = birth_second
    if birth_longitude is not None:
        profile["birth_longitude"] = birth_longitude

    return profile


def build_base_context(user_data: dict) -> dict:
    return {
        "stress": user_data.get("stress", 0.5),
        "urgency": user_data.get("urgency", 0.5),
    }


def slots_to_context(base: dict, slots: Optional[dict]) -> dict:
    return {**(base or {}), **(slots or {})}


def build_reading_reply(
    user_data: dict,
    active_text: str,
    known_slots: Optional[dict],
) -> Tuple[str, dict, dict]:
    profile = build_user_profile(user_data)
    context = slots_to_context(build_base_context(user_data), known_slots)

    motif_id = user_data.get("ritual_motif_id", "静かなる光")

    result = oracle_engine.predict(
        user_profile=profile,
        context_feats=context,
        user_text=active_text,
        motif_id=motif_id,
    )

    reply_text = result.get("message", "……識の声は、いまはまだ遠いようです。")

    if not user_data.get("birth_date"):
        reply_text += "\n\n……生まれた日の刻印を預ければ、より深く観測できるでしょう。"

    if user_data.get("birth_date") and user_data.get("birth_hour") in (None, ""):
        reply_text += "\n\n……生まれた時刻までわかれば、観測はさらに深く澄みます。"

    return reply_text, result, context


# -------------------------
# ヘルスチェック
# -------------------------
@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "app": "SHIKI LINE Bot",
        "time": datetime.now(timezone.utc).isoformat(),
        "model": GEMINI_MODEL,
    }


# -------------------------
# LINE Callback
# -------------------------
@app.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature")
    body = await request.body()

    if not signature:
        raise HTTPException(status_code=400, detail="Missing X-Line-Signature")

    try:
        handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        logger.warning("Invalid LINE signature")
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.exception("Callback handling error")
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse({"ok": True})


# -------------------------
# メッセージ受信
# -------------------------
@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event: MessageEvent):
    u_id = event.source.user_id
    user_text = normalize_text(event.message.text)

    if not user_text:
        try:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(
                    text="……ことばがまだ届いていないようです。もう一度、静かに問いを送ってください。"
                )
            )
        except Exception:
            logger.exception("Empty text reply failed")
        return

    try:
        user_ref = db.collection("users").document(u_id)
        snap = user_ref.get()
        user_data = snap.to_dict() or {}

        reply, result, ctx = build_reading_reply(user_data, user_text, {})

        update_payload = {
            "last_active": datetime.now(timezone.utc),
            "last_user_message": user_text,
            "last_oracle_message": reply,
            "last_oracle_summary": result.get("summary", {}),
            "updated_at": datetime.now(timezone.utc),
        }

        user_ref.set(update_payload, merge=True)

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply)
        )

    except Exception:
        logger.exception("handle_text_message error")

        fallback = "……識の視界が一時的に揺らぎました。少しだけ間を置いて、もう一度問いかけてください。"

        try:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=fallback)
            )
        except Exception:
            logger.exception("Fallback reply failed")


# -------------------------
# エントリポイント
# -------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
