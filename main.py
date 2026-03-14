import os
import json
import logging
import threading
import random
import re
import unicodedata
from datetime import datetime, timezone
from typing import Optional, List, Dict

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from linebot import LineBotApi, WebhookHandler
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, 
    QuickReply, QuickReplyButton, PostbackAction, PostbackEvent,
    DatetimePickerAction, TemplateSendMessage, ButtonsTemplate
)
from linebot.exceptions import InvalidSignatureError
from google import genai
import firebase_admin
from firebase_admin import credentials, firestore
from oracle_engine import OracleEngine

# 初期化
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s : %(message)s")
logger = logging.getLogger(__name__)
app = FastAPI()

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
    firebase_admin.initialize_app(credentials.Certificate(key_dict))
db = firestore.client()

oracle_engine = OracleEngine(gemini_client=genai_client, model_name=CHAT_MODEL)

# 40種のモチーフ
ALL_MOTIFS = [
    "銀の鍵", "砂時計", "古びた鏡", "聖なる滴", "琥珀の蝶", "折れた剣", "青い月", "羅針盤", "封じられた手紙", "金の天秤",
    "揺れる灯火", "水晶の髑髏", "沈黙の鐘", "茨の冠", "無垢な羽根", "双子の蛇", "星の砂", "錆びた歯車", "輝く聖杯", "黒い薔薇",
    "白い孔雀", "秘められた林檎", "古の地図", "時の歯車", "深海の真珠", "暁の鳥", "黄昏の雫", "不滅の炎", "氷の心臓", "奏でる竪琴",
    "空の器", "叡智の梟", "秘密の鍵穴", "約束の指輪", "流れる雲", "不動の岩", "踊る影", "導きの杖", "遮られた眼", "語らぬ仮面"
]

def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", (text or "").strip())

# メイン返信ロジック
def process_and_push_reply(user_id: str, user_text: str, motif_label: Optional[str] = None, selected_date: Optional[str] = None, selected_time: Optional[str] = None) -> None:
    try:
        user_ref = db.collection("users").document(user_id)
        user_data = user_ref.get().to_dict() or {}
        user_name = user_data.get("name")

        # リセット
        if user_text == "リセット":
            user_ref.delete()
            line_bot_api.push_message(user_id, TextSendMessage(text="ようこそ、探究者の方。新たな観測を始めましょう。まずは、あなた様をどのようにお呼びすればよろしいですか？"))
            return

        # 0. 名前取得
        if not user_name:
            user_ref.set({"name": user_text}, merge=True)
            send_birthday_picker(user_id, f"……{user_text}様ですね。心に刻みました。次に、あなたの生まれた日を教えてください。")
            return

        # 1. 生年月日登録（省略：既存導線を維持）
        if not user_data.get("birth_date"):
            if selected_date:
                user_ref.set({"birth_date": selected_date}, merge=True)
                send_time_picker(user_id)
            else: send_birthday_picker(user_id, f"{user_name}様。生まれた日を教えてください。")
            return

        if user_data.get("birth_hour") is None:
            if selected_time: user_ref.set({"birth_hour": int(selected_time.split(":")[0])}, merge=True); line_bot_api.push_message(user_id, TextSendMessage(text="刻印が完成しました。今、あなたが視たいことは何でしょうか。"))
            elif user_text == "UNKNOWN_TIME": user_ref.set({"birth_hour": 12}, merge=True); line_bot_api.push_message(user_id, TextSendMessage(text="承知いたしました。では、今あなたが視たいことは何でしょうか。"))
            else: send_time_picker(user_id)
            return

        # 2. 深掘り & 鑑定フロー
        if not user_data.get("is_dialogue_mode") and not user_data.get("pending_consult"):
            if len(user_text) <= 5: # 短い単語への問い返し
                user_ref.update({"temp_category": user_text})
                line_bot_api.push_message(user_id, TextSendMessage(text=f"……「{user_text}」についてですね。その奥にある想いを、もう少しだけ詳しく教えていただけますか？"))
                return
            else: # 十分な情報があればモチーフ提示
                user_ref.update({"pending_consult": user_text})
                sampled = random.sample(ALL_MOTIFS, 4)
                buttons = [QuickReplyButton(action=PostbackAction(label=m, data=f"action=select_motif&label={m}", display_text=m)) for m in sampled]
                line_bot_api.push_message(user_id, TextSendMessage(text="準備は整いました。心に触れる象徴を一つ選んでください。", quick_reply=QuickReply(items=buttons)))
                return

        # 3. 鑑定実行（モチーフ選択後）
        if motif_label:
            profile = {"name": user_name, "birth_year": int(user_data["birth_date"].split("-")[0]), "birth_month": int(user_data["birth_date"].split("-")[1]), "birth_day": int(user_data["birth_date"].split("-")[2]), "birth_hour": user_data["birth_hour"]}
            consult_text = user_data.get("pending_consult", "これからの運勢")
            mood = user_data.get("temp_category", "")
            result = oracle_engine.predict(profile, consult_text, motif_label, is_dialogue=False, current_mood=mood)
            user_ref.update({"is_dialogue_mode": True, "chat_history": f"識の神託: {result['message']}\n", "last_motif": motif_label, "dialogue_count": 1})
            line_bot_api.push_message(user_id, TextSendMessage(text=result["message"]))
            return

        # 4. 対話モード
        if user_data.get("is_dialogue_mode"):
            history = user_data.get("chat_history", "")
            count = user_data.get("dialogue_count", 1)
            
            # 終了確認後の分岐
            close_keywords = ["はい", "お願いします", "さようなら", "大丈夫です", "ありがとう", "解決しました"]
            if any(k in user_text for k in close_keywords) and "淵へ戻って" in history:
                user_ref.update({"is_dialogue_mode": False, "chat_history": "", "temp_category": firestore.DELETE_FIELD, "pending_consult": firestore.DELETE_FIELD})
                line_bot_api.push_message(user_id, TextSendMessage(text=f"……承知いたしました。{user_name}様、あなたの歩みに幸いがあらんことを。"))
                return

            profile = {"name": user_name, "birth_year": int(user_data["birth_date"].split("-")[0])}
            result = oracle_engine.predict(profile, user_text, user_data.get("last_motif"), is_dialogue=True, chat_history=history)
            user_ref.update({"chat_history": history + f"あなた: {user_text}\n識: {result['message']}\n", "dialogue_count": count + 1})
            line_bot_api.push_message(user_id, TextSendMessage(text=result["message"]))
            return

    except Exception as e:
        logger.exception("Error")
        line_bot_api.push_message(user_id, TextSendMessage(text="識の視界が揺らぎました。"))

# （UI補助関数、コールバック部分は前回のコードと同じです）
