import os
import json
import logging
from datetime import datetime, timezone

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from linebot.exceptions import InvalidSignatureError

import google.generativeai as genai

import firebase_admin
from firebase_admin import credentials, firestore

from oracle_engine import EngineState, OracleEngine


# -------------------------
# ログ設定
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s : %(message)s"
)
logger = logging.getLogger(__name__)


# -------------------------
# FastAPI
# -------------------------
app = FastAPI(title="SHIKI LINE Bot")


# -------------------------
# 環境変数
# -------------------------
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
FIREBASE_SERVICE_ACCOUNT_JSON = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")

missing = []
if not LINE_CHANNEL_ACCESS_TOKEN:
    missing.append("LINE_CHANNEL_ACCESS_TOKEN")
if not LINE_CHANNEL_SECRET:
    missing.append("LINE_CHANNEL_SECRET")
if not GEMINI_API_KEY:
    missing.append("GEMINI_API_KEY")
if not FIREBASE_SERVICE_ACCOUNT_JSON:
    missing.append("FIREBASE_SERVICE_ACCOUNT_JSON")

if missing:
    raise RuntimeError(f"必須環境変数が未設定です: {', '.join(missing)}")


# -------------------------
# LINE 初期化
# -------------------------
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)


# -------------------------
# Gemini 初期化
# -------------------------
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)


# -------------------------
# Oracle Engine 初期化
# -------------------------
oracle_state = EngineState()
oracle_engine = OracleEngine(oracle_state)


# -------------------------
# Firebase 初期化
# -------------------------
if not firebase_admin._apps:
    try:
        key_dict = json.loads(FIREBASE_SERVICE_ACCOUNT_JSON)

        logger.info("Firebase project_id: %s", key_dict.get("project_id"))
        logger.info("Firebase client_email: %s", key_dict.get("client_email"))
        logger.info("Firebase private_key_id: %s", key_dict.get("private_key_id"))
        logger.info(
            "Firebase private_key format OK: %s",
            str(key_dict.get("private_key", "")).startswith("-----BEGIN PRIVATE KEY-----")
        )

        cred = credentials.Certificate(key_dict)
        firebase_admin.initialize_app(cred)
        logger.info("Firebase initialization successful using env JSON.")
    except Exception as e:
        logger.exception("Firebase initialization failed")
        raise RuntimeError(f"Firebase initialization failed: {e}")

db = firestore.client()


# -------------------------
# Gemini 応答生成（朝通知用）
# -------------------------
def generate_mystical_message(user_text: str) -> str:
    prompt = (
        "あなたは神秘的な存在『識（SHIKI）』。孤独を肯定し、静かに寄り添います。"
        f"\nユーザーの昨日の言葉：『{user_text}』"
        "\nこの言葉を元に、今日を歩むための占い的な一言を80文字以内で作成してください。"
        "\n言い回しが毎回同じにならないようにしてください。"
        "\n語尾に必ず『――識より』を添えて。"
    )

    try:
        response = model.generate_content(prompt)
        text = getattr(response, "text", None)
        if text and text.strip():
            return text.strip()
        return "新しい朝が来ました。そのままのあなたで。――識より"
    except Exception as e:
        logger.error("Gemini morning error: %s", e)
        return "新しい朝が来ました。そのままのあなたで。――識より"


# -------------------------
# API
# -------------------------
@app.get("/")
def root():
    return {
        "status": "online",
        "message": "SHIKI System is running."
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "firebase_initialized": bool(firebase_admin._apps),
        "gemini_model": GEMINI_MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/morning-push")
def morning_push():
    try:
        users_ref = db.collection("users").stream()
        count = 0

        for user in users_ref:
            u_id = user.id
            u_data = user.to_dict() or {}
            last_msg = u_data.get("last_msg", "静かな心")

            msg_text = generate_mystical_message(last_msg)
            line_bot_api.push_message(u_id, TextSendMessage(text=msg_text))
            count += 1

        logger.info("Morning push completed. sent_count=%s", count)
        return {"status": "completed", "sent_count": count}

    except Exception as e:
        logger.exception("Push error")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature")
    body = await request.body()

    if not signature:
        raise HTTPException(status_code=400, detail="Missing X-Line-Signature header")

    body_text = body.decode("utf-8")

    try:
        handler.handle(body_text, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.exception("Webhook handling error")
        raise HTTPException(status_code=500, detail=f"Webhook handling failed: {e}")

    return "OK"


@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    try:
        u_id = event.source.user_id
        u_text = event.message.text.strip()

        user_ref = db.collection("users").document(u_id)
        user_doc = user_ref.get()
        user_data = user_doc.to_dict() or {}

        # 1. 発言を保存
        user_ref.set(
            {
                "last_msg": u_text,
                "last_active": datetime.now(timezone.utc)
            },
            merge=True
        )

        # 2. pending があれば「質問への回答」とみなして神託を返す
        pending_question = user_data.get("pending_question")
        pending_topic = user_data.get("pending_topic")
        pending_context = user_data.get("pending_context")
        pending_original_text = user_data.get("pending_original_text")

        if pending_question and pending_topic and pending_context and pending_original_text:
            user_profile = {
                "birth_month": int(user_data.get("birth_month", 6)),
                "resilience": float(user_data.get("resilience", 0.55)),
                "sensitivity": float(user_data.get("sensitivity", 0.70)),
                "patience": float(user_data.get("patience", 0.45))
            }

            memory = {
                "repeat_count": int(user_data.get("repeat_count", 1)),
                "volatility": float(user_data.get("volatility", 0.55))
            }

            updated_context = oracle_engine.apply_observation_answer(
                context_feats=pending_context,
                answer_text=u_text
            )

            oracle_result = oracle_engine.predict(
                user_profile=user_profile,
                context_feats=updated_context,
                user_text=pending_original_text,
                date_str=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                horizon="today",
                memory=memory
            )

            reply_text = oracle_result["message"]

            user_ref.set(
                {
                    "pending_question": firestore.DELETE_FIELD,
                    "pending_topic": firestore.DELETE_FIELD,
                    "pending_context": firestore.DELETE_FIELD,
                    "pending_original_text": firestore.DELETE_FIELD,
                    "last_topic": oracle_result["topic"],
                    "last_oracle_message": oracle_result["message"],
                    "oracle_engine_version": oracle_result["engine_version"],
                    "last_observation_answer": u_text
                },
                merge=True
            )

            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=reply_text)
            )
            return

        # 3. 通常時はまず質問を1つ返す
        detected_topic = oracle_engine.topic_classifier.classify(u_text)

        base_context = {
            "stress": float(user_data.get("stress", 0.60)),
            "sleep_deficit": float(user_data.get("sleep_deficit", 0.50)),
            "loneliness": float(user_data.get("loneliness", 0.55)),
            "urgency": float(user_data.get("urgency", 0.65))
        }

        followup_questions = oracle_engine.question_engine.get_followup_questions(detected_topic)
        first_question = followup_questions[0]

        user_ref.set(
            {
                "pending_question": first_question,
                "pending_topic": detected_topic,
                "pending_context": base_context,
                "pending_original_text": u_text
            },
            merge=True
        )

        reply_text = (
            "観測をもう少しだけ深めます。\n"
            f"{first_question}"
        )

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply_text)
        )

    except Exception as e:
        logger.exception("handle_text error")
        try:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="識の観測にわずかな乱れが生じました。少し時間を置いてもう一度声をかけてください。")
            )
        except Exception:
            logger.exception("Reply fallback failed")
