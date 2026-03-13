import os
import json
import logging
import unicodedata
import threading
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from linebot import LineBotApi, WebhookHandler
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, 
    QuickReply, QuickReplyButton, MessageAction, PostbackAction
)
from linebot.exceptions import InvalidSignatureError
from google import genai
import firebase_admin
from firebase_admin import credentials, firestore
from oracle_engine import OracleEngine

# -------------------------
# 初期化
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s : %(message)s")
logger = logging.getLogger(__name__)
app = FastAPI(title="SHIKI LINE Bot")

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FIREBASE_SERVICE_ACCOUNT_JSON = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-1.5-flash")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
genai_client = genai.Client(api_key=GEMINI_API_KEY)

if not firebase_admin._apps:
    key_dict = json.loads(FIREBASE_SERVICE_ACCOUNT_JSON)
    cred = credentials.Certificate(key_dict)
    firebase_admin.initialize_app(cred)
db = firestore.client()

oracle_engine = OracleEngine(gemini_client=genai_client, model_name=CHAT_MODEL)

# -------------------------
# 儀式用モチーフ定義
# -------------------------
MOTIFS = [
    {"label": "銀の鍵", "id": "silver_key"},
    {"label": "砂時計", "id": "hourglass"},
    {"label": "古びた鏡", "id": "ancient_mirror"},
    {"label": "聖なる滴", "id": "holy_drop"}
]

# -------------------------
# ユーティリティ
# -------------------------
def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", (text or "").strip())

def build_user_profile(user_data: dict) -> dict:
    bd = user_data.get("birth_date", "1990-01-01")
    y, m, d = map(int, bd.split("-"))
    profile = {"birth_year": y, "birth_month": m, "birth_day": d}
    for key in ["birth_hour", "birth_minute"]:
        if user_data.get(key) is not None: profile[key] = int(user_data[key])
    return profile

# -------------------------
# 返信ロジック
# -------------------------
def process_and_push_reply(user_id: str, user_text: str, motif_id: Optional[str] = None) -> None:
    try:
        user_ref = db.collection("users").document(user_id)
        user_data = user_ref.get().to_dict() or {}

        # リセット処理
        if user_text == "リセット":
            user_ref.delete()
            line_bot_api.push_message(user_id, TextSendMessage(text="……すべての記録を虚空へ返しました。初めから観測を始めましょう。あなたの生まれた日はいつですか？（例：1990-01-01）"))
            return

        # 生年月日がない場合、入力を促す
        if not user_data.get("birth_date"):
            if re.match(r"^\d{4}-\d{2}-\d{2}$", user_text):
                user_ref.set({"birth_date": user_text}, merge=True)
                line_bot_api.push_message(user_id, TextSendMessage(text=f"……{user_text}。あなたの刻印を受け取りました。次に、今あなたが一番視たいことを教えてください。"))
            else:
                line_bot_api.push_message(user_id, TextSendMessage(text="……ようこそ。観測を始める前に、あなたの生まれた日を教えてください。（例：1992-05-15）"))
            return

        # モチーフ選択の提示（儀式フェーズ）
        if not motif_id:
            # 相談内容を一時保存
            user_ref.set({"pending_consult": user_text}, merge=True)
            
            # クイックリプライボタン作成
            buttons = [QuickReplyButton(action=PostbackAction(label=m["label"], data=f"action=select_motif&id={m['id']}", display_text=m["label"])) for m in MOTIFS]
            quick_reply = QuickReply(items=buttons)
            
            line_bot_api.push_message(user_id, TextSendMessage(text="……準備は整いました。最後に、あなたの直感を重ねます。いま、心に触れる象徴を一つ選んでください。", quick_reply=quick_reply))
            return

        # 鑑定実行（モチーフ選択後）
        profile = build_user_profile(user_data)
        consult_text = user_data.get("pending_consult", user_text)
        result = oracle_engine.predict(profile, {"stress": 0.5}, consult_text, motif_id)
        
        reply = result.get("message", "……識の声が途切れました。")
        line_bot_api.push_message(user_id, TextSendMessage(text=reply))
        
        # 鑑定後は一時データをクリア
        user_ref.update({"pending_consult": firestore.DELETE_FIELD})

    except Exception:
        logger.exception("process_and_push_reply error")
        line_bot_api.push_message(user_id, TextSendMessage(text="……識の視界が揺らぎました。少し間を置いてください。"))

# -------------------------
# LINE Callback & Event Handler
# -------------------------
@app.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature")
    body = await request.body()
    try:
        handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400)
    return JSONResponse({"ok": True})

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event: MessageEvent):
    user_id = event.source.user_id
    user_text = normalize_text(event.message.text)
    if user_text:
        threading.Thread(target=process_and_push_reply, args=(user_id, user_text), daemon=True).start()

from linebot.models import PostbackEvent
@handler.add(PostbackEvent)
def handle_postback(event: PostbackEvent):
    user_id = event.source.user_id
    data = dict(x.split('=') for x in event.postback.data.split('&'))
    
    if data.get("action") == "select_motif":
        motif_id = data.get("id")
        # 非同期で鑑定実行へ飛ばす（user_textは空で、内部でpending_consultを使用）
        threading.Thread(target=process_and_push_reply, args=(user_id, "", motif_id), daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
