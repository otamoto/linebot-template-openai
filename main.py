import os
import json
import logging
import sys
from datetime import datetime, timezone

# --- ログ設定 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s : %(message)s"
)
logger = logging.getLogger(__name__)

# --- パス設定 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- ライブラリのインポート ---
try:
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import JSONResponse
    from linebot import LineBotApi, WebhookHandler
    from linebot.models import MessageEvent, TextMessage, TextSendMessage
    from linebot.exceptions import InvalidSignatureError
    import google.generativeai as genai
    import firebase_admin
    from firebase_admin import credentials, firestore
except ImportError as e:
    logger.error(f"必須ライブラリが不足しています: {e}")
    raise

# --- アプリ初期化 ---
app = FastAPI()

# --- 環境変数の取得 ---
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FIREBASE_SERVICE_ACCOUNT_JSON = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# --- 必須環境変数チェック ---
missing_vars = []
if not LINE_CHANNEL_ACCESS_TOKEN:
    missing_vars.append("LINE_CHANNEL_ACCESS_TOKEN")
if not LINE_CHANNEL_SECRET:
    missing_vars.append("LINE_CHANNEL_SECRET")
if not GEMINI_API_KEY:
    missing_vars.append("GEMINI_API_KEY")
if not FIREBASE_SERVICE_ACCOUNT_JSON:
    missing_vars.append("FIREBASE_SERVICE_ACCOUNT_JSON")

if missing_vars:
    msg = f"必須環境変数が未設定です: {', '.join(missing_vars)}"
    logger.error(msg)
    raise RuntimeError(msg)

# --- LINE / Gemini 初期化 ---
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)

# --- Firebase初期化（env JSON方式） ---
if not firebase_admin._apps:
    try:
        key_dict = json.loads(FIREBASE_SERVICE_ACCOUNT_JSON)

        # 安全な確認ログ（秘密鍵全文は出さない）
        logger.info(f"Firebase project_id: {key_dict.get('project_id')}")
        logger.info(f"Firebase client_email: {key_dict.get('client_email')}")
        logger.info(f"Firebase private_key_id: {key_dict.get('private_key_id')}")
        logger.info(
            "Firebase private_key format OK: %s",
            str(key_dict.get("private_key", "")).startswith("-----BEGIN PRIVATE KEY-----")
        )

        cred = credentials.Certificate(key_dict)
        firebase_admin.initialize_app(cred)
        logger.info("Firebase initialization successful using env JSON.")

    except json.JSONDecodeError as e:
        logger.error(f"FIREBASE_SERVICE_ACCOUNT_JSON のJSON解析に失敗: {e}")
        raise
    except Exception as e:
        logger.error(f"Firebase Initialization Failed: {e}")
        raise

db = firestore.client()

# --- Gemini応答生成 ---
def generate_shiki_reply(user_text: str) -> str:
    """
    LINEで話しかけられたときの返信を生成
    """
    prompt = (
        "あなたは神秘的な存在『識（SHIKI）』です。"
        "孤独を否定せず、静かに寄り添い、少しだけ神秘的に答えてください。"
        "説教はしません。"
        "返答は1〜3文、140文字以内、日本語で自然に。"
        "最後に毎回『――識より』を付けてください。"
        f"\n\nユーザーの言葉: {user_text}"
    )
    try:
        response = model.generate_content(prompt)
        text = getattr(response, "text", None)
        if text and text.strip():
            return text.strip()
        return "その言葉は、夜の水面に静かに沈みました。――識より"
    except Exception as e:
        logger.error(f"Gemini reply error: {e}")
        return "その言葉は、夜の水面に静かに沈みました。――識より"

def generate_mystical_message(user_text: str) -> str:
    """
    朝のプッシュ用メッセージ生成
    """
    prompt = (
        "あなたは神秘的な存在『識（SHIKI）』。孤独を肯定し、静かに寄り添います。"
        f"\nユーザーの昨日の言葉：『{user_text}』"
        "\nこの言葉を元に、今日を歩むための占い的な一言を80文字以内で作成してください。"
        "\n語尾に必ず「――識より」を添えて。"
    )
    try:
        response = model.generate_content(prompt)
        text = getattr(response, "text", None)
        if text and text.strip():
            return text.strip()
        return "新しい朝が来ました。そのままのあなたで。――識より"
    except Exception as e:
        logger.error(f"Gemini morning error: {e}")
        return "新しい朝が来ました。そのままのあなたで。――識より"

# --- エンドポイント ---

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
    """
    Firestore の users コレクション全員に朝メッセージ送信
    """
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

        logger.info(f"Morning push completed. sent_count={count}")
        return {"status": "completed", "sent_count": count}

    except Exception as e:
        logger.error(f"Push Error: {e}")
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

    try:
        handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.error(f"Webhook handling error: {e}")
        raise HTTPException(status_code=500, detail="Webhook handling failed")

    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    """
    ユーザーの発言を保存し、Geminiで個別返信する
    """
    try:
        u_id = event.source.user_id
        u_text = event.message.text

        # 発言保存
        db.collection("users").document(u_id).set(
            {
                "last_msg": u_text,
                "last_active": datetime.now(timezone.utc)
            },
            merge=True
        )

        # 個別返信生成
        reply_text = generate_shiki_reply(u_text)

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply_text)
        )

    except Exception as e:
        logger.error(f"handle_text error: {e}")
        try:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="少しだけ、識の声が乱れました。もう一度試してください。")
            )
        except Exception as reply_err:
            logger.error(f"Reply fallback failed: {reply_err}")
