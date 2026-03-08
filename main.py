import os
import json
import logging
import sys
from datetime import datetime

# パス設定：同じフォルダのファイルを確実に見つけるための処理
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from linebot.exceptions import InvalidSignatureError
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore

# インポート：oracle_engine.pyからクラスを読み込む
try:
    from oracle_engine import OracleEngine, EngineState
except ImportError:
    # 実行環境によるパスの違いを吸収
    from .oracle_engine import OracleEngine, EngineState

# --- 初期設定 ---
app = FastAPI()
logging.basicConfig(level=logging.INFO)

# HerokuのConfig Varsから取得
LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
FIREBASE_CONFIG_STR = os.getenv('FIREBASE_CONFIG')

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Firebase初期化
if not firebase_admin._apps:
    try:
        cred_dict = json.loads(FIREBASE_CONFIG_STR)
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        logging.error(f"Firebase Init Error: {e}")

db = firestore.client()

# --- ロジック ---

def generate_mystical_message(user_text):
    """Geminiを使って寄り添うメッセージを生成"""
    prompt = (
        f"あなたは『識（SHIKI）』という名の、孤独を肯定し、静かに寄り添う存在です。"
        f"昨日のユーザーの言葉：『{user_text}』"
        f"この言葉を受けて、今日という日を静かに始めるためのメッセージを80文字以内で作成してください。"
        f"占い的な示唆を含め、最後に「――識より」と添えて。"
    )
    try:
        response = model.generate_content(prompt)
        return response.text
    except:
        return "新しい朝が来ました。あなたのままで、今日を歩んでください。――識より"

# --- エンドポイント ---

@app.get("/")
def root():
    return {"message": "SHIKI System is online."}

@app.get("/morning-push")
def morning_push():
    """朝のプッシュ通知を実行するURL"""
    try:
        # is_premium が true のユーザーを取得
        users = db.collection('users').where('is_premium', '==', True).stream()
        
        count = 0
        # エンジンの初期化（n=2000のシミュレーション等）
        engine = OracleEngine(EngineState())

        for user in users:
            u_id = user.id
            u_data = user.to_dict()
            last_msg = u_data.get('last_msg', "平穏な日々")
            
            # メッセージ生成
            msg_text = generate_mystical_message(last_msg)
            
            # LINE送信
            line_bot_api.push_message(u_id, TextSendMessage(text=msg_text))
            count += 1
            
        return {"status": "completed", "sent_count": count}
    except Exception as e:
        logging.error(f"Morning Push Error: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/callback")
async def callback(request: Request):
    """LINEからのメッセージを受け取る口"""
    signature = request.headers.get('X-Line-Signature')
    body = await request.body()
    try:
        handler.handle(body.decode('utf-8'), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    """ユーザーからの話しかけに反応し、Firebaseを更新する"""
    u_id = event.source.user_id
    u_text = event.message.text
    
    # ユーザーの最新発言を保存（明日の朝のプッシュに使用）
    db.collection('users').document(u_id).set({
        'last_msg': u_text,
        'last_active': datetime.now()
    }, merge=True)
    
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text="あなたの言葉は、識の奥底に刻まれました。明日の朝、またお話ししましょう。")
    )
