import os
import json
import logging
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from linebot.exceptions import InvalidSignatureError
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore

# --- ログ設定 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- アプリ初期化 ---
app = FastAPI()

# --- 環境変数の取得（HerokuのConfig Varsから読み込む） ---
LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
FIREBASE_CONFIG_JSON = os.getenv('FIREBASE_CONFIG')

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- Firebase初期化（ファイルを介さずメモリ上で読み込む） ---
if not firebase_admin._apps:
    try:
        # Herokuから受け取ったJSON文字列を辞書形式に変換
        cred_dict = json.loads(FIREBASE_CONFIG_JSON, strict=False)
        # 秘密鍵の改行文字が壊れないよう補正
        if 'private_key' in cred_dict:
            cred_dict['private_key'] = cred_dict['private_key'].replace('\\n', '\n')
            
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        logger.info("Firebase initialization successful.")
    except Exception as e:
        logger.error(f"Firebase Initialization Failed: {e}")

db = firestore.client()

# --- 識（SHIKI）のメインロジック ---
@app.get("/")
def root():
    return {"status": "online", "message": "SHIKI system has been restored safely."}

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
    
    # 返信をFirebaseに保存
    db.collection('users').document(u_id).set({
        'last_msg': u_text,
        'last_active': datetime.now()
    }, merge=True)
    
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text="あなたの言葉は、識の奥底に静かに刻まれました。")
    )
