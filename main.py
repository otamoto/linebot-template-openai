import os
import json
import logging
import sys
from datetime import datetime

# --- ログ設定 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- パス設定：自作モジュールを確実に見つける ---
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

# --- 自作エンジンのインポート ---
try:
    from oracle_engine import OracleEngine, EngineState
except ImportError:
    try:
        from .oracle_engine import OracleEngine, EngineState
    except ImportError as e:
        logger.error(f"oracle_engine.py が見つかりません: {e}")

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

# --- Firebase初期化（秘密鍵の改行エラーを強制修正） ---
if not firebase_admin._apps:
    try:
        if not FIREBASE_CONFIG_JSON:
            raise ValueError("環境変数 FIREBASE_CONFIG が空です。")
        
        # JSONをパース
        cred_dict = json.loads(FIREBASE_CONFIG_JSON, strict=False)
        
        # 【重要】Herokuで壊れやすい秘密鍵の改行コード（\n）を正常な形に置換
        if 'private_key' in cred_dict:
            cred_dict['private_key'] = cred_dict['private_key'].replace('\\n', '\n')
            
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        logger.info("Firebase initialization successful.")
    except Exception as e:
        logger.error(f"Firebase Initialization Failed: {e}")
        # 起動を止めてログに詳細を残す
        raise

db = firestore.client()

# --- メッセージ生成ロジック ---
def generate_mystical_message(user_text):
    prompt = (
        f"あなたは『識（SHIKI）』。孤独を肯定し、静かに寄り添う存在です。\n"
        f"ユーザーの昨日の言葉：『{user_text}』\n"
        f"この言葉を受け止め、今日という静かな始まりに相応しい、占い的な示唆を含むメッセージを80文字以内で作成してください。\n"
        f"語尾に必ず「――識より」を添えて。"
    )
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Gemini Error: {e}")
        return "新しい朝が来ました。そのままのあなたで、今日を歩んでください。――識より"

# --- エンドポイント ---

@app.get("/")
def root():
    """サーバーの生存確認用"""
    return {"status": "online", "service": "SHIKI System", "timestamp": str(datetime.now())}

@app.get("/morning-push")
def morning_push():
    """手動またはSchedulerでプッシュ通知を送るURL"""
    try:
        # Firebaseから is_premium == True のユーザーを検索
        users_ref = db.collection('users').where('is_premium', '==', True).stream()
        
        count = 0
        for user in users_ref:
            u_id = user.id
            u_data = user.to_dict()
            last_msg = u_data.get('last_msg', "新しい朝")
            
            # メッセージ生成
            msg_text = generate_mystical_message(last_msg)
            
            # LINE送信
            line_bot_api.push_message(u_id, TextSendMessage(text=msg_text))
            count += 1
            
        return {"status": "completed", "sent_count": count}
    except Exception as e:
        logger.error(f"Push Notification Error: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/callback")
async def callback(request: Request):
    """LINE Messaging APIのWebhook受付"""
    signature = request.headers.get('X-Line-Signature')
    body = await request.body()
    try:
        handler.handle(body.decode('utf-8'), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    """ユーザーからの返信をFirebaseに保存"""
    u_id = event.source.user_id
    u_text = event.message.text
    
    # ユーザーデータを更新（merge=Trueで既存データを壊さない）
    db.collection('users').document(u_id).set({
        'last_msg': u_text,
        'last_active': datetime.now()
    }, merge=True)
    
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text="あなたの言葉は、識の奥底に静かに刻まれました。")
    )
