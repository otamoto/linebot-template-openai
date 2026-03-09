import os
import json
import logging
import sys
from datetime import datetime

# --- ログ設定 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- パス設定 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- インポート ---
from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from linebot.exceptions import InvalidSignatureError
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore

# --- 初期設定 ---
app = FastAPI()

LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
FIREBASE_CONFIG_JSON = os.getenv('FIREBASE_CONFIG')

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- Firebase初期化（安全な環境変数読み込み版） ---
if not firebase_admin._apps:
    try:
        if not FIREBASE_CONFIG_JSON:
            raise ValueError("FIREBASE_CONFIG is missing in Heroku Config Vars")
        
        # JSONとして読み込み、秘密鍵の改行エラーを補正
        cred_dict = json.loads(FIREBASE_CONFIG_JSON, strict=False)
        if 'private_key' in cred_dict:
            cred_dict['private_key'] = cred_dict['private_key'].replace('\\n', '\n')
            
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        logger.info("Firebase connection established.")
    except Exception as e:
        logger.error(f"Firebase Error: {e}")
        # 起動を止めないための暫定処理（エラーログは残す）

db = firestore.client()

# --- ロジック ---
def generate_mystical_message(user_text):
    prompt = f"あなたは『識（SHIKI）』。ユーザーの言葉『{user_text}』に寄り添う、80文字以内の占い的メッセージを。最後に「――識より」と付けて。"
    try:
        response = model.generate_content(prompt)
        return response.text
    except:
        return "新しい朝が来ました。あなたのままで。――識より"

@app.get("/")
def root():
    return {"status": "online", "message": "SHIKI System is restored."}

@app.get("/morning-push")
def morning_push():
    try:
        users = db.collection('users').where('is_premium', '==', True).stream()
        count = 0
        for user in users:
            u_id = user.id
            last_msg = user.to_dict().get('last_msg', "平穏")
            msg_text = generate_mystical_message(last_msg)
            line_bot_api.push_message(u_id, TextSendMessage(text=msg_text))
            count += 1
        return {"status": "completed", "sent_count": count}
    except Exception as e:
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
    db.collection('users').document(u_id).set({
        'last_msg': event.message.text,
        'last_active': datetime.now()
    }, merge=True)
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text="識に刻まれました。"))
