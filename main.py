import os
import json
import logging
import re
import unicodedata
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
# 生年月日パース
# -------------------------
def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text.strip())


def parse_japanese_era_date(text: str):
    """
    昭和 / 平成 / 令和 を西暦へ変換
    """
    text = normalize_text(text)
    era_map = {
        "昭和": 1925,  # 昭和1年 = 1926
        "平成": 1988,  # 平成1年 = 1989
        "令和": 2018   # 令和1年 = 2019
    }

    m = re.search(r"(昭和|平成|令和)\s*(元|\d+)\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", text)
    if not m:
        return None

    era_name = m.group(1)
    era_year_raw = m.group(2)
    month = int(m.group(3))
    day = int(m.group(4))

    era_year = 1 if era_year_raw == "元" else int(era_year_raw)
    year = era_map[era_name] + era_year

    try:
        dt = datetime(year, month, day)
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None


def parse_birth_date(text: str):
    """
    幅広い生年月日入力を YYYY-MM-DD に変換する
    受け付ける例:
    - 1990-05-12
    - 1990/5/12
    - 1990年5月12日
    - 19900512
    - 1990512
    - 昭和60年5月12日
    - 平成3年11月2日
    - 令和2年6月10日
    """
    text = normalize_text(text)

    # 和暦
    era_result = parse_japanese_era_date(text)
    if era_result:
        return era_result

    # YYYY-MM-DD / YYYY/MM/DD / YYYY.MM.DD
    m = re.match(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$", text)
    if m:
        year, month, day = map(int, m.groups())
        try:
            dt = datetime(year, month, day)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None

    # YYYY年M月D日
    m = re.match(r"^(\d{4})年(\d{1,2})月(\d{1,2})日$", text)
    if m:
        year, month, day = map(int, m.groups())
        try:
            dt = datetime(year, month, day)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None

    # 8桁 YYYYMMDD
    m = re.match(r"^(\d{4})(\d{2})(\d{2})$", text)
    if m:
        year, month, day = map(int, m.groups())
        try:
            dt = datetime(year, month, day)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None

    # 7桁 YYYYMDD または YYYYMM D
    m = re.match(r"^(\d{4})(\d{1,2})(\d{1,2})$", text)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        day = int(m.group(3))
        try:
            dt = datetime(year, month, day)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None

    return None


def looks_like_birth_date(text: str) -> bool:
    return parse_birth_date(text) is not None


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

        # 発言保存
        user_ref.set(
            {
                "last_msg": u_text,
                "last_active": datetime.now(timezone.utc)
            },
            merge=True
        )

        # --------------------------------
        # 1. 生年月日入力の受け取り
        # --------------------------------
        if looks_like_birth_date(u_text):
            parsed_birth = parse_birth_date(u_text)
            if parsed_birth:
                user_ref.set(
                    {
                        "birth_date": parsed_birth
                    },
                    merge=True
                )

                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(
                        text=(
                            f"生まれた日の気配を受け取りました。\n"
                            f"{parsed_birth} として記録しておきます。"
                        )
                    )
                )
                return

        # --------------------------------
        # 2. 状態取得
        # --------------------------------
        conversation_state = user_data.get("conversation_state", "idle")
        pending_questions = user_data.get("pending_questions", [])
        question_step = int(user_data.get("question_step", 0))
        question_answers = user_data.get("question_answers", [])
        pending_original_text = user_data.get("pending_original_text")
        current_topic = user_data.get("current_topic")
        pending_context = user_data.get("pending_context")
        is_paid = bool(user_data.get("is_paid", False))

        # --------------------------------
        # 3. 質問フロー中なら回答を受ける
        # --------------------------------
        if conversation_state == "awaiting_answers" and pending_questions and pending_original_text and current_topic and pending_context:
            question_answers.append(u_text)
            question_step += 1

            # まだ次の質問がある
            if question_step < len(pending_questions):
                next_question = pending_questions[question_step]

                user_ref.set(
                    {
                        "question_step": question_step,
                        "question_answers": question_answers
                    },
                    merge=True
                )

                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text=next_question)
                )
                return

            # 質問が終わったので神託生成
            user_profile = {
                "birth_month": 6,
                "resilience": float(user_data.get("resilience", 0.55)),
                "sensitivity": float(user_data.get("sensitivity", 0.70)),
                "patience": float(user_data.get("patience", 0.45))
            }

            birth_date = user_data.get("birth_date")
            if birth_date:
                try:
                    user_profile["birth_month"] = int(birth_date.split("-")[1])
                except Exception:
                    pass

            memory = {
                "repeat_count": int(user_data.get("repeat_count", 1)),
                "volatility": float(user_data.get("volatility", 0.55))
            }

            updated_context = dict(pending_context)
            for ans in question_answers:
                updated_context = oracle_engine.apply_observation_answer(updated_context, ans)

            horizon = "week" if is_paid else "today"

            oracle_result = oracle_engine.predict(
                user_profile=user_profile,
                context_feats=updated_context,
                user_text=pending_original_text,
                date_str=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                horizon=horizon,
                memory=memory,
                is_paid=is_paid
            )

            reply_text = oracle_result["message"]

            # 生年月日未登録なら自然に誘導
            if not birth_date:
                reply_text += (
                    "\n\nもっと深く流れを視るには、生まれた日の気配も重ねた方が精度が上がります。"
                    "\n生年月日は、1990-05-12、1990年5月12日、昭和60年5月12日 のような形で送ってもらえれば大丈夫です。"
                )

            # 状態クリア
            user_ref.set(
                {
                    "conversation_state": "completed_reading",
                    "pending_questions": firestore.DELETE_FIELD,
                    "question_step": firestore.DELETE_FIELD,
                    "question_answers": firestore.DELETE_FIELD,
                    "pending_original_text": firestore.DELETE_FIELD,
                    "current_topic": firestore.DELETE_FIELD,
                    "pending_context": firestore.DELETE_FIELD,
                    "last_topic": oracle_result["topic"],
                    "last_oracle_message": oracle_result["message"],
                    "oracle_engine_version": oracle_result["engine_version"]
                },
                merge=True
            )

            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text=reply_text)
            )
            return

        # --------------------------------
        # 4. 通常時：質問フロー開始
        # --------------------------------
        detected_topic = oracle_engine.topic_classifier.classify(u_text)

        base_context = {
            "stress": float(user_data.get("stress", 0.60)),
            "sleep_deficit": float(user_data.get("sleep_deficit", 0.50)),
            "loneliness": float(user_data.get("loneliness", 0.55)),
            "urgency": float(user_data.get("urgency", 0.65))
        }

        question_set = oracle_engine.question_engine.get_question_set(
            topic=detected_topic,
            is_paid=is_paid
        )

        first_question = question_set[0]

        user_ref.set(
            {
                "conversation_state": "awaiting_answers",
                "pending_questions": question_set,
                "question_step": 0,
                "question_answers": [],
                "pending_original_text": u_text,
                "current_topic": detected_topic,
                "pending_context": base_context
            },
            merge=True
        )

        opening = "少しだけ深く視るために、いくつか聞かせてください。"
        if is_paid:
            opening = "では、もう少し深いところまで流れを視ます。順に答えてください。"

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=f"{opening}\n{first_question}")
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
