import os
import json
import logging
import sys
from datetime import datetime

# --- ログ設定 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- パス設定：同じフォルダのファイルを確実に見つける ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- ライブラリのインポート ---
try:
    from fastapi import FastAPI, Request, HTTPException
    from linebot import LineBotApi, WebhookHandler
    from linebot.models import MessageEvent, TextMessage, TextSendMessage
    from linebot.exceptions import InvalidSignatureError
    import google.generativeai as genai
    import firebase_admin
    from firebase_admin import credentials, firestore
except ImportError as e:
    logger.error(f"Required library missing: {e}")
    raise

# --- 自作エンジンのインポート ---
try:
    from oracle_engine import OracleEngine, EngineState
except ImportError:
    try:
        from .oracle_engine import OracleEngine, EngineState
    except ImportError as e:
        logger.error(f"oracle_engine.py not found: {e}")

# --- アプリ初期化 ---
app = FastAPI()

# --- 環境変数の取得 ---
LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
FIREBASE_CONFIG_JSON = os.getenv('FIREBASE_CONFIG')

# インスタンス化
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- Firebase初期化（JSONパースエラー対策済み） ---
if not firebase_admin._apps:
    try:
        if not FIREBASE_CONFIG_JSON:
            raise ValueError("Environment variable FIREBASE_CONFIG is empty.")
        
        # strict=False にすることで、コピー時の改行コードの乱れを許容して読み込む
        cred_dict = json.loads(FIREBASE_CONFIG_JSON, strict=False)
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        logger.info("Firebase initialization successful.")
    except Exception as e:
        logger.error(f"Firebase Initialization Failed: {e}")
        # ここで止まらないようにダミーをセットする場合もありますが、今回はエラーログを優先
        raise

db = firestore.client()

# --- メッセージ生成ロジック ---
def generate_mystical_message(user_text):
    prompt = (
        f"あなたは『識（SHIKI）』。孤独を肯定し、静かに寄り添う存在です。\n"
        f"昨日の言葉：『{user_text}』\n"
        f"今日を歩むための占い的な一言を80文字以内で作成してください。語尾に「――識より」を付けて。"
    )
    try:
        response = model.generate_content(prompt)
        return response.text
    except:
        return "新しい朝が来ました。そのままのあなたで。――識より"

# --- エンドポイント ---

@app.get("/")
def root():
    # サーバーが生きているか確認するためのURL
    return {"message": "SHIKI System is Online."}

@app.get("/morning-push")
def morning_push():
    try:
        # is_premium が true のユーザーを取得
        users_ref = db.collection('users').where('is_premium', '==', True).stream()
        
        count = 0
        for user in users_ref:
            u_id = user.id
            u_data = user.to_dict()
            last_msg = u_data.get('last_msg', "静かな心")
            
            # 送信
            msg_text = generate_mystical_message(last_msg)
            line_bot_api.push_message(u_id, TextSendMessage(text=msg_text))
            count += 1
            
        return {"status": "completed", "sent_count": count}
    except Exception as e:
        logger.error(f"Push Error: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/callback")
async def callback(request: Request):
    signature = request.headers.get('X-Line-Signature')
    body = await request.body()
    try:
        handler.handle(body.decode('utf-8'), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    u_id = event.source.user_id
    u_text = event.message.text
    # ユーザー発言を更新
    db.collection('users').document(u_id).set({
        'last_msg': u_text,
        'last_active': datetime.now()
    }, merge=True)
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text="あなたの言葉は、識の奥底へ。"))
