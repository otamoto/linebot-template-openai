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

# --- 初期設定 ---
app = FastAPI()
logging.basicConfig(level=logging.INFO)

# HerokuのConfig Varsから取得
LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
FIREBASE_CONFIG = os.getenv('FIREBASE_CONFIG')

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Firebase初期化
if not firebase_admin._apps:
    cred_dict = json.loads(FIREBASE_CONFIG)
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)
db = firestore.client()

# oracle_engine.pyからインポート
from oracle_engine import OracleEngine, EngineState

# --- ヘルパー関数 ---
def generate_mystical_message(user_text):
    prompt = f"あなたは『識』。朝、孤独を肯定し寄り添う言葉を。昨日ユーザーは『{user_text}』と言っていました。100文字以内で作成せよ。"
    return model.generate_content(prompt).text

# --- エンドポイント ---

@app.get("/") # これを追加することで URL 単体でアクセスしてもエラーにならなくなります
def root():
    return {"message": "SHIKI System is running."}

@app.get("/morning-push") # ここが morning-push の入り口です
def morning_push():
    # is_premium が true のユーザーを探す
    users = db.collection('users').where('is_premium', '==', True).stream()
    
    count = 0
    for user in users:
        u_id = user.id
        u_data = user.to_dict()
        last_msg = u_data.get('last_msg', "新しい始まり")
        
        msg_text = generate_mystical_message(last_msg)
        try:
            line_bot_api.push_message(u_id, TextSendMessage(text=msg_text))
            count += 1
        except Exception as e:
            logging.error(f"Error: {e}")
            
    return {"status": "completed", "sent_count": count}

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
    # 対話ロジック（必要に応じて追加）
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text="あなたの声は届いています。"))
