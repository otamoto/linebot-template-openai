import os
import json
import logging
import sys
from datetime import datetime

# --- ログ設定 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- パス設定：自作モジュールや鍵ファイルを確実に見つける ---
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
    logger.error(f"必須ライブラリが不足しています: {e}")
    raise

# --- アプリ初期化 ---
app = FastAPI()

# --- 環境変数の取得 ---
LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# インスタンス化
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- Firebase初期化（ファイルを直接読み込む方式） ---
if not firebase_admin._apps:
    try:
        key_path = os.path.join(current_dir, 'firebase-key.json')
        if not os.path.exists(key_path):
            raise FileNotFoundError(f"{key_path} が見つかりません。")
            
        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred)
        logger.info("Firebase initialization successful using local file.")
    except Exception as e:
        logger.error(f"Firebase Initialization Failed: {e}")
        raise

db = firestore.client()

# --- 占いメッセージ生成ロジック ---
def generate_mystical_message(user_text):
    """昨日の言葉を元に、Geminiが占い的な示唆を生成する"""
    prompt = (
        f"あなたは神秘的な存在『識（SHIKI）』。孤独を肯定し、静かに寄り添います。\n"
        f"ユーザーの昨日の言葉：『{user_text}』\n"
        f"この言葉を元に、今日を歩むための占い的な一言を80文字以内で作成してください。\n"
        f"語尾に必ず「――識より」を添えて。"
    )
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Gemini Error: {e}")
        return "新しい朝が来ました。そのままのあなたで。――識より"

# --- エンドポイント ---

@app.get("/")
def root():
    """生存確認用"""
    return {"status": "online", "message": "SHIKI System is running."}

@app.get("/morning-push")
def morning_push():
    """
    【占い実行】
    データベース（Firestore）の 'users' コレクションにいる
    全ユーザーに対して個別の占いメッセージをプッシュ通知します。
    """
    try:
        # 全ユーザーを取得
        users_ref = db.collection('users').stream()
        
        count = 0
        for user in users_ref:
            u_id = user.id  # LINEのユーザーID
            u_data = user.to_dict()
            last_msg = u_data.get('last_msg', "静かな心")
            
            # 占い生成
            msg_text = generate_mystical_message(last_msg)
            
            # LINEプッシュ送信
            line_bot_api.push_message(u_id, TextSendMessage(text=msg_text))
            count += 1
            
        return {"status": "completed", "sent_count": count}
    except Exception as e:
        logger.error(f"Push Error: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/callback")
async def callback(request: Request):
    """LINE Webhook用"""
    signature = request.headers.get('X-Line-Signature')
    body = await request.body()
    try:
        handler.handle(body.decode('utf-8'), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    """ユーザーの言葉を記憶する"""
    u_id = event.source.user_id
    u_text = event.message.text
    
    # Firebaseの 'users' コレクションに保存（上書きマージ）
    db.collection('users').document(u_id).set({
        'last_msg': u_text,
        'last_active': datetime.now()
    }, merge=True)
    
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text="あなたの言葉は、識の奥底に静かに刻まれました。")
    )
