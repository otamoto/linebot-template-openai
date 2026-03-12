import os
import json
import logging
import re
import threading
import unicodedata
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from linebot import LineBotApi, WebhookHandler
from linebot.models import (
    MessageEvent,
    TextMessage,
    TextSendMessage,
    PostbackEvent,
    QuickReply,
    QuickReplyButton,
    MessageAction,
    DatetimePickerAction,
    PostbackAction,
)
from linebot.exceptions import InvalidSignatureError

import google.generativeai as genai

import firebase_admin
from firebase_admin import credentials, firestore

from oracle_engine import EngineState, OracleEngine


# -------------------------
# ログ設定
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s : %(message)s"
)
logger = logging.getLogger(__name__)


# -------------------------
# FastAPI
# -------------------------
app = FastAPI(title="SHIKI LINE Bot")


# -------------------------
# 環境変数
# -------------------------
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
FIREBASE_SERVICE_ACCOUNT_JSON = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")

missing = []
if not LINE_CHANNEL_ACCESS_TOKEN:
    missing.append("LINE_CHANNEL_ACCESS_TOKEN")
if not LINE_CHANNEL_SECRET:
    missing.append("LINE_CHANNEL_SECRET")
if not GEMINI_API_KEY:
    missing.append("GEMINI_API_KEY")
if not FIREBASE_SERVICE_ACCOUNT_JSON:
    missing.append("FIREBASE_SERVICE_ACCOUNT_JSON")

if missing:
    raise RuntimeError(f"必須環境変数が未設定です: {', '.join(missing)}")


# -------------------------
# LINE 初期化
# -------------------------
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)


# -------------------------
# Gemini 初期化
# -------------------------
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)


# -------------------------
# Oracle Engine 初期化
# -------------------------
oracle_state = EngineState()
oracle_engine = OracleEngine(oracle_state)


# -------------------------
# Firebase 初期化
# -------------------------
if not firebase_admin._apps:
    try:
        key_dict = json.loads(FIREBASE_SERVICE_ACCOUNT_JSON)
        cred = credentials.Certificate(key_dict)
        firebase_admin.initialize_app(cred)
        logger.info("Firebase initialization successful using env JSON.")
    except Exception as e:
        logger.exception("Firebase initialization failed")
        raise RuntimeError(f"Firebase initialization failed: {e}")

db = firestore.client()


# -------------------------
# Persona 制約
# -------------------------
PERSONA_GUARDRAIL = """
あなたは神秘的な未来観測者『識（SHIKI）』です。
次の制約を最上位で守ってください。

- 絶対的断定をしない
- 説教しない
- 安易な同情をしない
- 命令しない
- 「絶対うまくいく」「こうすべき」などと言わない
- 直前の神託文をそのまま繰り返さない
- 一般的なカウンセラーやコンサルタントのような話し方に寄りすぎない
- 短く自然に、しかし識らしい静かな距離感を保つ
""".strip()


# -------------------------
# 共通ユーティリティ
# -------------------------
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text.strip())


def parse_japanese_era_date(text: str) -> Optional[str]:
    text = normalize_text(text)
    era_map = {
        "昭和": 1925,
        "平成": 1988,
        "令和": 2018
    }

    m = re.search(r"(昭和|平成|令和)\s*(元|\d+)\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", text)
    if not m:
        return None

    era_name = m.group(1)
    era_year_raw = m.group(2)
    month = int(m.group(3))
    day = int(m.group(4))

    era_year = 1 if era_year_raw == "元" else int(era_year_raw)
    year = era_map[era_name] + era_year

    try:
        dt = datetime(year, month, day)
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None


def parse_birth_date(text: str) -> Optional[str]:
    text = normalize_text(text)

    era_result = parse_japanese_era_date(text)
    if era_result:
        return era_result

    m = re.match(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$", text)
    if m:
        try:
            dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None

    m = re.match(r"^(\d{4})年(\d{1,2})月(\d{1,2})日$", text)
    if m:
        try:
            dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None

    m = re.match(r"^(\d{4})(\d{2})(\d{2})$", text)
    if m:
        try:
            dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None

    m = re.match(r"^(\d{4})(\d{1,2})(\d{1,2})$", text)
    if m:
        try:
            dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None

    return None


def looks_like_birth_date(text: str) -> bool:
    return parse_birth_date(text) is not None


def is_short_ambiguous_utterance(text: str) -> bool:
    t = normalize_text(text)
    short_set = {
        "ん？", "え？", "えっと", "うーん", "なるほど", "そうなんだ",
        "そうか", "ふむ", "んー", "うーむ", "へえ", "ほう"
    }
    return len(t) <= 8 and t in short_set


def is_resume_like_message(text: str) -> bool:
    t = normalize_text(text)
    keywords = ["続き", "その後", "前回", "あれから", "変化", "まだ", "やっぱり"]
    return any(k in t for k in keywords)


def is_new_consult_message(text: str) -> bool:
    t = normalize_text(text)
    keywords = ["それとは別に", "ちなみに", "別件", "他にも", "仕事のことも", "家族のことも", "別の相談"]
    return any(k in t for k in keywords)


def is_disagree_message(text: str) -> bool:
    t = normalize_text(text)
    keywords = ["違う気がする", "違う", "そこじゃない", "当たってない", "そういうことじゃない", "ズレてる"]
    return any(k in t for k in keywords)


def is_clarify_message(text: str) -> bool:
    t = normalize_text(text)
    keywords = ["どういうこと", "わかりやすく", "つまり", "意味", "簡単に"]
    return any(k in t for k in keywords)


def is_action_message(text: str) -> bool:
    t = normalize_text(text)
    keywords = ["どうしたら", "何をすれば", "結局", "連絡していい", "待てばいい", "行動"]
    return any(k in t for k in keywords)


def is_deepen_message(text: str) -> bool:
    t = normalize_text(text)
    keywords = ["もっと詳しく", "深く", "相手の気持ち", "今週", "先も", "もっと見て"]
    return any(k in t for k in keywords)


def required_slots_for_topic(topic: str, plan_tier: str) -> list[str]:
    common = ["time_continuity", "emotion", "desired_action"]
    if topic == "love":
        slots = common + ["relationship_distance"]
    elif topic == "work":
        slots = common + ["main_stressor"]
    else:
        slots = common + ["person_type"]

    if plan_tier in {"paid", "deep"}:
        return slots[:4]
    return slots[:3]


def merge_known_slots(existing: dict, new_data: dict) -> dict:
    merged = dict(existing or {})
    for k, v in (new_data or {}).items():
        if v is not None and v != "":
            merged[k] = v
    return merged


def build_user_profile(user_data: dict) -> dict:
    birth_year, birth_month, birth_day = 1990, 6, 15
    birth_date = user_data.get("birth_date")
    if birth_date:
        try:
            y, m, d = birth_date.split("-")
            birth_year, birth_month, birth_day = int(y), int(m), int(d)
        except Exception:
            pass

    return {
        "birth_year": birth_year,
        "birth_month": birth_month,
        "birth_day": birth_day,
        "resilience": float(user_data.get("resilience", 0.55)),
        "sensitivity": float(user_data.get("sensitivity", 0.70)),
        "patience": float(user_data.get("patience", 0.45))
    }


def build_memory(user_data: dict) -> dict:
    return {
        "repeat_count": int(user_data.get("repeat_count", 1)),
        "volatility": float(user_data.get("volatility", 0.55))
    }


def build_base_context(user_data: dict) -> dict:
    return {
        "stress": float(user_data.get("stress", 0.60)),
        "sleep_deficit": float(user_data.get("sleep_deficit", 0.50)),
        "loneliness": float(user_data.get("loneliness", 0.55)),
        "urgency": float(user_data.get("urgency", 0.65))
    }


def slots_to_context(base_context: dict, known_slots: dict) -> dict:
    updated = dict(base_context or {})
    known_slots = known_slots or {}

    emotion = str(known_slots.get("emotion", ""))
    continuity = str(known_slots.get("time_continuity", ""))
    desired_action = str(known_slots.get("desired_action", ""))
    relationship_distance = str(known_slots.get("relationship_distance", ""))
    main_stressor = str(known_slots.get("main_stressor", ""))

    if any(x in emotion for x in ["不安", "焦", "怖", "こわ"]):
        updated["urgency"] = min(float(updated.get("urgency", 0.5)) + 0.15, 1.0)
    if any(x in emotion for x in ["悲", "寂", "孤独", "つらい"]):
        updated["loneliness"] = min(float(updated.get("loneliness", 0.5)) + 0.15, 1.0)
    if any(x in emotion for x in ["怒", "イライラ", "腹立"]):
        updated["stress"] = min(float(updated.get("stress", 0.5)) + 0.15, 1.0)
    if emotion == "__UNRESOLVED__":
        updated["urgency"] = min(float(updated.get("urgency", 0.5)) + 0.05, 1.0)

    if any(x in continuity for x in ["前から", "ずっと", "長く", "続いて"]):
        updated["stress"] = min(float(updated.get("stress", 0.5)) + 0.05, 1.0)
    if continuity == "__UNRESOLVED__":
        updated["stress"] = min(float(updated.get("stress", 0.5)) + 0.03, 1.0)

    if any(x in desired_action for x in ["待", "様子を見"]):
        updated["urgency"] = max(float(updated.get("urgency", 0.5)) - 0.10, 0.0)
    if any(x in desired_action for x in ["動きたい", "連絡", "伝えたい"]):
        updated["urgency"] = min(float(updated.get("urgency", 0.5)) + 0.10, 1.0)
    if desired_action == "__UNRESOLVED__":
        updated["urgency"] = min(float(updated.get("urgency", 0.5)) + 0.03, 1.0)

    if any(x in relationship_distance for x in ["離", "既読無視", "返事がない", "遠"]):
        updated["loneliness"] = min(float(updated.get("loneliness", 0.5)) + 0.10, 1.0)

    if any(x in main_stressor for x in ["仕事量", "忙", "残業"]):
        updated["stress"] = min(float(updated.get("stress", 0.5)) + 0.12, 1.0)
    if any(x in main_stressor for x in ["人間関係", "上司", "同僚"]):
        updated["stress"] = min(float(updated.get("stress", 0.5)) + 0.10, 1.0)

    return updated


def extract_slots_from_text(text: str, topic: str) -> dict:
    t = normalize_text(text)
    slots = {}

    if any(x in t for x in ["前から", "ずっと", "長く", "しばらく", "続いて"]):
        slots["time_continuity"] = "前から続いている"
    elif any(x in t for x in ["急に", "突然", "いきなり", "最近急"]):
        slots["time_continuity"] = "急に強くなった"

    if any(x in t for x in ["不安", "焦", "こわ", "怖"]):
        slots["emotion"] = "不安が強い"
    elif any(x in t for x in ["悲しい", "寂しい", "孤独", "つらい"]):
        slots["emotion"] = "悲しさや寂しさが強い"
    elif any(x in t for x in ["怒", "イライラ", "腹立"]):
        slots["emotion"] = "怒りや苛立ちが強い"
    elif any(x in t for x in ["何も感じない", "空虚", "虚しい"]):
        slots["emotion"] = "空虚さが強い"

    if any(x in t for x in ["待ちたい", "様子を見たい", "少し置きたい", "待つ"]):
        slots["desired_action"] = "待ちたい"
    elif any(x in t for x in ["動きたい", "連絡したい", "伝えたい", "進めたい"]):
        slots["desired_action"] = "動きたい"
    elif any(x in t for x in ["終わらせたい", "離れたい"]):
        slots["desired_action"] = "終わらせたい"

    if topic == "love":
        if any(x in t for x in ["既読無視", "返事がない", "離れてる", "距離がある", "かなり離れて"]):
            slots["relationship_distance"] = "少し離れている"
        elif any(x in t for x in ["近い", "会えてる", "普通に話せる"]):
            slots["relationship_distance"] = "近い"

    elif topic == "work":
        if any(x in t for x in ["上司", "同僚", "人間関係"]):
            slots["main_stressor"] = "人間関係"
        elif any(x in t for x in ["仕事量", "忙しい", "残業", "業務量"]):
            slots["main_stressor"] = "仕事量"
        elif any(x in t for x in ["将来", "先が見えない", "不安"]):
            slots["main_stressor"] = "将来の不安"

    else:
        if any(x in t for x in ["家族", "親", "母", "父"]):
            slots["person_type"] = "家族"
        elif any(x in t for x in ["友達", "友人"]):
            slots["person_type"] = "友人"
        elif any(x in t for x in ["職場", "上司", "同僚"]):
            slots["person_type"] = "職場"
        elif any(x in t for x in ["彼", "彼女", "恋人", "パートナー"]):
            slots["person_type"] = "恋人"

    return slots


def safe_generate(prompt: str, fallback: str) -> str:
    try:
        response = model.generate_content(prompt)
        text = getattr(response, "text", "").strip()
        return text if text else fallback
    except Exception:
        logger.exception("Gemini generation failed")
        return fallback


# -------------------------
# Quick Reply builders
# -------------------------
def make_quick_reply(items: list[QuickReplyButton]) -> QuickReply:
    return QuickReply(items=items)


def build_quick_reply_for_slot(slot_name: str, topic: str = "") -> Optional[QuickReply]:
    if slot_name == "emotion":
        items = [
            QuickReplyButton(action=MessageAction(label="不安", text="不安")),
            QuickReplyButton(action=MessageAction(label="悲しさ", text="悲しさ")),
            QuickReplyButton(action=MessageAction(label="イライラ", text="イライラ")),
            QuickReplyButton(action=MessageAction(label="空虚さ", text="空虚さ")),
            QuickReplyButton(action=PostbackAction(label="言葉にできない", data="slot=emotion&value=__UNRESOLVED__", display_text="言葉にできない")),
        ]
        return make_quick_reply(items)

    if slot_name == "desired_action":
        items = [
            QuickReplyButton(action=MessageAction(label="動きたい", text="動きたい")),
            QuickReplyButton(action=MessageAction(label="様子を見たい", text="様子を見たい")),
            QuickReplyButton(action=MessageAction(label="終わらせたい", text="終わらせたい")),
            QuickReplyButton(action=PostbackAction(label="まだ決めきれない", data="slot=desired_action&value=__UNRESOLVED__", display_text="まだ決めきれない")),
        ]
        return make_quick_reply(items)

    if slot_name == "time_continuity":
        items = [
            QuickReplyButton(action=MessageAction(label="急に強くなった", text="急に強くなった")),
            QuickReplyButton(action=MessageAction(label="前から続いている", text="前から続いている")),
            QuickReplyButton(action=PostbackAction(label="うまく分けられない", data="slot=time_continuity&value=__UNRESOLVED__", display_text="うまく分けられない")),
        ]
        return make_quick_reply(items)

    if slot_name == "relationship_distance":
        items = [
            QuickReplyButton(action=MessageAction(label="近い", text="近い")),
            QuickReplyButton(action=MessageAction(label="少し離れている", text="少し離れている")),
            QuickReplyButton(action=MessageAction(label="かなり離れている", text="かなり離れている")),
        ]
        return make_quick_reply(items)

    if slot_name == "main_stressor":
        items = [
            QuickReplyButton(action=MessageAction(label="仕事量", text="仕事量")),
            QuickReplyButton(action=MessageAction(label="人間関係", text="人間関係")),
            QuickReplyButton(action=MessageAction(label="将来の不安", text="将来の不安")),
        ]
        return make_quick_reply(items)

    if slot_name == "person_type":
        items = [
            QuickReplyButton(action=MessageAction(label="家族", text="家族")),
            QuickReplyButton(action=MessageAction(label="友人", text="友人")),
            QuickReplyButton(action=MessageAction(label="職場", text="職場")),
            QuickReplyButton(action=MessageAction(label="恋人", text="恋人")),
        ]
        return make_quick_reply(items)

    return None


def build_birthdate_quick_reply() -> QuickReply:
    items = [
        QuickReplyButton(
            action=DatetimePickerAction(
                label="日付を選ぶ",
                data="type=birthdate_picker",
                mode="date",
                initial="1990-01-01",
                min="1940-01-01",
                max=now_utc().strftime("%Y-%m-%d"),
            )
        ),
        QuickReplyButton(
            action=PostbackAction(
                label="自分で入力する",
                data="type=birthdate_manual",
                display_text="自分で入力する"
            )
        ),
        QuickReplyButton(
            action=PostbackAction(
                label="今は送らない",
                data="type=birthdate_skip",
                display_text="今は送らない"
            )
        ),
    ]
    return make_quick_reply(items)


def should_offer_quick_reply(user_data: dict, missing_slots: list[str]) -> bool:
    if not missing_slots:
        return False

    ui_state = user_data.get("ui_state", "none")
    if ui_state != "none":
        return False

    last_quick_reply_context = user_data.get("last_quick_reply_context") or {}
    last_slot = last_quick_reply_context.get("slot")
    if last_slot == missing_slots[0]:
        return False

    return True


def build_slot_prompt(slot_name: str, topic: str) -> str:
    mapping = {
        "emotion": "今の気持ちにいちばん近いものを、ここから選んでください。",
        "desired_action": "言葉にしづらければ、近い方を選んでください。",
        "time_continuity": "この揺れは、どちらに近いですか。",
        "relationship_distance": "相手との距離は今どうですか。",
        "main_stressor": "いちばんしんどいのはどれですか。",
        "person_type": "その相手は誰に近いですか。",
    }
    return mapping.get(slot_name, "近いものがあれば、ここから選んでください。")


def ui_state_name_for_slot(slot_name: str) -> str:
    return f"awaiting_{slot_name}"


# -------------------------
# Router
# -------------------------
def route_user_message(user_text: str, user_data: dict) -> str:
    mode = user_data.get("conversation_mode", "idle")

    if looks_like_birth_date(user_text):
        return "birthdate_update"

    if mode == "post_oracle":
        if is_disagree_message(user_text):
            return "disagree_oracle"
        if is_action_message(user_text):
            return "post_oracle_action"
        if is_deepen_message(user_text):
            return "post_oracle_deepen"
        if is_clarify_message(user_text):
            return "post_oracle_explain"
        if is_new_consult_message(user_text):
            return "new_consult_reset"
        if is_short_ambiguous_utterance(user_text):
            return "post_oracle_explain"
        return "post_oracle_explain"

    if mode == "suspended":
        if is_new_consult_message(user_text):
            return "new_consult_reset"
        if user_data.get("bookmark"):
            return "resume_session"
        if is_short_ambiguous_utterance(user_text):
            return "ambient_response"
        return "start_consult"

    if mode == "consulting":
        if is_new_consult_message(user_text):
            return "new_consult_reset"
        if is_short_ambiguous_utterance(user_text):
            return "continue_consult"
        return "continue_consult"

    if is_short_ambiguous_utterance(user_text):
        return "ambient_response"

    return "start_consult"


# -------------------------
# Implicit Suspension
# -------------------------
def build_bookmark_payload(user_data: dict) -> dict:
    last_oracle_summary = user_data.get("last_oracle_summary") or {}
    current_topic = user_data.get("current_topic") or user_data.get("last_topic") or "relationship"
    known_slots = user_data.get("known_slots") or {}

    next_observation_point = ""
    if current_topic == "love":
        next_observation_point = known_slots.get("relationship_distance", "相手との距離感がどう動くか")
    elif current_topic == "work":
        next_observation_point = known_slots.get("main_stressor", "仕事の重さがどう変わるか")
    else:
        next_observation_point = known_slots.get("person_type", "相手との距離感がどう動くか")

    oracle_summary = last_oracle_summary.get("core_meaning", "今は流れを急がず見直す時期")
    core_issue = last_oracle_summary.get("risk_hint", "判断を急ぎやすいこと")
    resume_prompt = (
        f"……再同期します。前回、あなたは『{oracle_summary}』という流れの中にいました。"
        f"その後、{next_observation_point}に変化はありましたか。"
    )

    return {
        "oracle_summary": oracle_summary,
        "core_issue": core_issue,
        "next_observation_point": next_observation_point,
        "resume_prompt": resume_prompt,
        "session_closed_at": now_utc().isoformat()
    }


def implicit_suspend_check(user_ref, user_data: dict) -> dict:
    mode = user_data.get("conversation_mode", "idle")
    if mode not in {"consulting", "post_oracle"}:
        return user_data

    last_active = user_data.get("last_active")
    if not last_active:
        return user_data

    try:
        if hasattr(last_active, "replace"):
            last_dt = last_active
        else:
            last_dt = datetime.fromisoformat(str(last_active))
    except Exception:
        return user_data

    if last_dt.tzinfo is None:
        last_dt = last_dt.replace(tzinfo=timezone.utc)

    if now_utc() - last_dt < timedelta(hours=24):
        return user_data

    bookmark = user_data.get("bookmark")
    if not bookmark:
        bookmark = build_bookmark_payload(user_data)

    patch = {
        "conversation_mode": "suspended",
        "bookmark": bookmark,
        "session_closed_at": now_utc()
    }
    user_ref.set(patch, merge=True)

    merged = dict(user_data)
    merged.update(patch)
    return merged


# -------------------------
# Background helper
# -------------------------
def fire_and_forget(target, *args, **kwargs):
    def runner():
        try:
            target(*args, **kwargs)
        except Exception:
            logger.exception("Background task failed")
    threading.Thread(target=runner, daemon=True).start()


def update_listener_snapshot_async(user_ref, user_text: str):
    t = normalize_text(user_text)
    summary = {
        "emotion_hint": "anxiety" if any(x in t for x in ["不安", "焦", "怖"]) else "",
        "fatigue_hint": "fatigue" if any(x in t for x in ["疲", "眠れて", "休めて"]) else "",
        "listener_snapshot_summary": f"latest_text_len={len(t)}"
    }
    user_ref.set({"listener_snapshot": summary}, merge=True)


# -------------------------
# Oracle / Integration
# -------------------------
def build_reading_reply(user_data: dict, active_text: str, known_slots: dict) -> Tuple[str, dict, dict]:
    plan_tier = user_data.get("plan_tier", "free")
    is_paid = plan_tier in {"paid", "deep"}
    horizon = "week" if is_paid else "today"

    base_context = user_data.get("last_context") or build_base_context(user_data)
    context_feats = slots_to_context(base_context, known_slots)
    user_profile = build_user_profile(user_data)
    memory = build_memory(user_data)

    oracle_result = oracle_engine.predict(
        user_profile=user_profile,
        context_feats=context_feats,
        user_text=active_text,
        horizon=horizon,
        memory=memory,
        is_paid=is_paid
    )

    reply_text = oracle_result["message"]

    if not user_data.get("birth_date"):
        reply_text += (
            "\n\n生まれた日の流れも重ねると、観測の精度が上がります。"
            "\nここで日付を選ぶこともできます。"
        )

    return reply_text, oracle_result, context_feats


def explain_oracle_simple(user_text: str, last_oracle_message: str, oracle_summary: dict) -> str:
    fallback = (
        f"簡単に言うと、{oracle_summary.get('core_meaning', '今は流れを急がず見直した方がいい時期です。')}"
        f"\n{oracle_summary.get('risk_hint', '')}"
    )

    prompt = f"""
{PERSONA_GUARDRAIL}

今回は新しい神託を作らず、直前の神託をわかりやすい日常語へ翻訳してください。
2〜4文程度。
神託文をそのまま再送しないでください。

直前の神託:
{last_oracle_message}

内部要約:
{json.dumps(oracle_summary, ensure_ascii=False)}

ユーザーの聞き方:
{user_text}
""".strip()

    return safe_generate(prompt, fallback)


def explain_oracle_action(user_text: str, last_oracle_message: str, oracle_summary: dict) -> str:
    fallback = (
        f"{oracle_summary.get('action_hint', '今は答えを急ぐより、ひとつだけ整えてから次を見る方がよさそうです。')}\n"
        f"避けるなら、{oracle_summary.get('risk_hint', '焦って決める動き')}です。"
    )

    prompt = f"""
{PERSONA_GUARDRAIL}

今回は新しい神託を作らず、直前の神託を行動レベルへ落としてください。
命令しないでください。
「今の流れでは〜の方が合っている」という言い方を優先してください。
2〜4文程度。

直前の神託:
{last_oracle_message}

内部要約:
{json.dumps(oracle_summary, ensure_ascii=False)}

ユーザーの聞き方:
{user_text}
""".strip()

    return safe_generate(prompt, fallback)


def handle_disagreement_repair() -> str:
    return "ズレを感じたなら、それは大事な反応です。どの部分がいちばん違うと感じたのか、そこだけ教えてください。"


def ambient_response(text: str) -> str:
    t = normalize_text(text)
    if t in {"ん？", "え？", "うーん", "えっと"}:
        return "急がなくて大丈夫です。そのまま、引っかかっているところを少しだけ言葉にしてみてください。"
    return "そのまま話してください。必要なところだけ、こちらで拾っていきます。"


def build_resume_reply(bookmark: dict) -> str:
    return bookmark.get(
        "resume_prompt",
        "……再同期します。前回の流れの続きを、もう一度見ていきましょう。"
    )


def unresolved_slot_response(slot_name: str) -> str:
    mapping = {
        "emotion": "無理に形にしなくても大丈夫です。濁ったまま、もう少し奥を見てみましょう。",
        "desired_action": "まだ決めきれなくても大丈夫です。その揺れごと観測していきます。",
        "time_continuity": "うまく分けられなくても大丈夫です。その曖昧さも手がかりになります。"
    }
    return mapping.get(slot_name, "はっきり分けられなくても大丈夫です。そのまま続けていきましょう。")


# -------------------------
# Event dispatcher helpers
# -------------------------
def get_or_create_user(user_id: str) -> Tuple[Any, dict]:
    user_ref = db.collection("users").document(user_id)
    user_doc = user_ref.get()
    user_data = user_doc.to_dict() or {}
    return user_ref, user_data


def apply_common_user_touch(user_ref, user_data: dict, last_msg: Optional[str] = None) -> dict:
    patch = {
        "last_active": now_utc(),
        "plan_tier": user_data.get("plan_tier", "free"),
    }
    if last_msg is not None:
        patch["last_msg"] = last_msg

    user_ref.set(patch, merge=True)
    merged = dict(user_data)
    merged.update(patch)
    return merged


# -------------------------
# Message event main flow
# -------------------------
def handle_message_flow(reply_token: str, user_ref, user_data: dict, user_text: str):
    # ui_state が残っていても、ユーザーがテキストで返した時点で会話優先
    if user_data.get("ui_state", "none") != "none":
        user_ref.set({"ui_state": "none", "last_quick_reply_context": firestore.DELETE_FIELD}, merge=True)
        user_data["ui_state"] = "none"
        user_data["last_quick_reply_context"] = None

    # 自然消滅チェックは current last_active 更新前の値で行うべきなので、
    # 呼び出し元で先に user_data を読み込み、この関数前に touch 済みでも old data から実行しないよう注意。
    route = route_user_message(user_text, user_data)
    mode = user_data.get("conversation_mode", "idle")

    if route == "birthdate_update":
        parsed_birth = parse_birth_date(user_text)
        if not parsed_birth:
            line_bot_api.reply_message(
                reply_token,
                TextSendMessage(text="その日付はまだうまく読み取れませんでした。西暦や和暦の形で、もう一度送ってみてください。")
            )
            return

        updated = {"birth_date": parsed_birth}
        user_ref.set(updated, merge=True)

        active_text = user_data.get("active_consultation_text") or user_data.get("last_consultation_text")
        current_topic = user_data.get("current_topic") or user_data.get("last_topic")
        known_slots = user_data.get("known_slots") or {}

        if active_text and current_topic:
            refreshed_user_data = {**user_data, **updated}
            reply_text, oracle_result, context_feats = build_reading_reply(
                user_data=refreshed_user_data,
                active_text=active_text,
                known_slots=known_slots
            )

            user_ref.set(
                {
                    "last_oracle_message": oracle_result["message"],
                    "last_oracle_summary": oracle_result["summary"],
                    "last_context": context_feats,
                    "conversation_mode": "post_oracle"
                },
                merge=True
            )

            line_bot_api.reply_message(
                reply_token,
                TextSendMessage(
                    text=(
                        f"生まれた日の気配を受け取りました。{parsed_birth} として記録しておきます。\n\n"
                        f"さっきの流れを、その巡りも重ねてもう一度視ました。\n{reply_text}"
                    )
                )
            )
            return

        line_bot_api.reply_message(
            reply_token,
            TextSendMessage(text=f"生まれた日の気配を受け取りました。{parsed_birth} として記録しておきます。前の内容が違っていた場合も、今回の内容で上書きされています。")
        )
        return

    if route == "resume_session":
        bookmark = user_data.get("bookmark") or {}
        user_ref.set({"conversation_mode": "consulting", "ui_state": "none"}, merge=True)
        line_bot_api.reply_message(reply_token, TextSendMessage(text=build_resume_reply(bookmark)))
        return

    if route == "new_consult_reset":
        user_ref.set(
            {
                "conversation_mode": "consulting",
                "current_topic": firestore.DELETE_FIELD,
                "active_consultation_text": firestore.DELETE_FIELD,
                "known_slots": {},
                "missing_slots": [],
                "last_question": firestore.DELETE_FIELD,
                "ui_state": "none",
                "last_quick_reply_context": firestore.DELETE_FIELD,
            },
            merge=True
        )
        user_data = {
            **user_data,
            "conversation_mode": "consulting",
            "current_topic": None,
            "active_consultation_text": None,
            "known_slots": {},
            "missing_slots": [],
            "last_question": None,
            "ui_state": "none",
        }
        route = "start_consult"

    if route == "ambient_response":
        line_bot_api.reply_message(reply_token, TextSendMessage(text=ambient_response(user_text)))
        return

    last_oracle_message = user_data.get("last_oracle_message", "")
    last_oracle_summary = user_data.get("last_oracle_summary") or {}

    if route == "post_oracle_explain":
        reply_text = explain_oracle_simple(user_text, last_oracle_message, last_oracle_summary)
        line_bot_api.reply_message(reply_token, TextSendMessage(text=reply_text))
        return

    if route == "post_oracle_action":
        reply_text = explain_oracle_action(user_text, last_oracle_message, last_oracle_summary)
        line_bot_api.reply_message(reply_token, TextSendMessage(text=reply_text))
        return

    if route == "post_oracle_deepen":
        current_topic = user_data.get("current_topic") or user_data.get("last_topic") or "relationship"
        known_slots = user_data.get("known_slots") or {}
        needed = required_slots_for_topic(current_topic, user_data.get("plan_tier", "free"))
        missing = [s for s in needed if not known_slots.get(s)]

        if missing and should_offer_quick_reply(user_data, missing):
            slot = missing[0]
            qr = build_quick_reply_for_slot(slot, current_topic)
            prompt = build_slot_prompt(slot, current_topic)
            user_ref.set(
                {
                    "conversation_mode": "consulting",
                    "last_question": prompt,
                    "ui_state": ui_state_name_for_slot(slot),
                    "last_quick_reply_context": {
                        "slot": slot,
                        "topic": current_topic,
                        "asked_at": now_utc().isoformat(),
                    },
                },
                merge=True
            )
            line_bot_api.reply_message(reply_token, TextSendMessage(text=f"では、さっきの流れをもう少し深く見ます。\n{prompt}", quick_reply=qr))
            return

        question_map = {
            "time_continuity": "その悩みって、急に強くなった感じですか？ それとも前からずっと続いていましたか？",
            "emotion": "今の気持ちにいちばん近いのはどれですか？ 不安 / 悲しさ / イライラ / 空虚さ",
            "desired_action": "今は動きたいですか？ それとも少し様子を見たい気持ちの方が近いですか？",
            "relationship_distance": "相手との距離は今どうですか？ 近い感じですか、それとも少し離れていますか？",
            "main_stressor": "いちばんしんどいのはどれに近いですか？ 仕事量 / 人間関係 / 将来の不安",
            "person_type": "その相手は誰に近いですか？ 家族 / 友人 / 職場 / 恋人"
        }
        next_question = question_map.get(missing[0], "もう少しだけ、今の引っかかりをそのまま話してみてください。") if missing else "もう少し深く見るために、いま一番引っかかっている部分をそのまま話してみてください。"

        user_ref.set(
            {
                "conversation_mode": "consulting",
                "last_question": next_question,
                "ui_state": "none",
            },
            merge=True
        )
        line_bot_api.reply_message(reply_token, TextSendMessage(text=f"では、さっきの流れをもう少し深く見ます。\n{next_question}"))
        return

    if route == "disagree_oracle":
        user_ref.set({"conversation_mode": "post_oracle", "repair_pending": True}, merge=True)
        line_bot_api.reply_message(reply_token, TextSendMessage(text=handle_disagreement_repair()))
        return

    if user_data.get("repair_pending") and mode == "post_oracle":
        known_slots = user_data.get("known_slots") or {}
        current_topic = user_data.get("current_topic") or user_data.get("last_topic") or "relationship"
        extracted = extract_slots_from_text(user_text, current_topic)
        known_slots = merge_known_slots(known_slots, extracted)

        user_ref.set(
            {
                "conversation_mode": "consulting",
                "repair_pending": False,
                "known_slots": known_slots,
                "ui_state": "none",
            },
            merge=True
        )
        user_data = {**user_data, "conversation_mode": "consulting", "repair_pending": False, "known_slots": known_slots}
        route = "continue_consult"

    if route == "start_consult":
        topic = oracle_engine.topic_classifier.classify(user_text)
        extracted = extract_slots_from_text(user_text, topic)
        known_slots = merge_known_slots({}, extracted)

        user_ref.set(
            {
                "conversation_mode": "consulting",
                "current_topic": topic,
                "active_consultation_text": user_text,
                "known_slots": known_slots,
                "last_consultation_text": user_text,
                "ui_state": "none",
            },
            merge=True
        )

        needed = required_slots_for_topic(topic, user_data.get("plan_tier", "free"))
        missing = [s for s in needed if not known_slots.get(s)]

        if len([k for k in needed if known_slots.get(k)]) >= min(3, len(needed)):
            bridge = "だいぶ輪郭が見えてきました。では、今のあなたに近い流れを言葉にします。"
            reply_text, oracle_result, context_feats = build_reading_reply(
                user_data={**user_data, "current_topic": topic},
                active_text=user_text,
                known_slots=known_slots
            )

            if not user_data.get("birth_date"):
                birth_qr = build_birthdate_quick_reply()
                user_ref.set(
                    {
                        "conversation_mode": "post_oracle",
                        "last_oracle_message": oracle_result["message"],
                        "last_oracle_summary": oracle_result["summary"],
                        "last_context": context_feats,
                        "known_slots": known_slots,
                        "last_topic": oracle_result["topic"],
                        "ui_state": "awaiting_birthdate",
                    },
                    merge=True
                )

                line_bot_api.reply_message(
                    reply_token,
                    TextSendMessage(text=f"{bridge}\n\n{reply_text}", quick_reply=birth_qr)
                )
                return

            user_ref.set(
                {
                    "conversation_mode": "post_oracle",
                    "last_oracle_message": oracle_result["message"],
                    "last_oracle_summary": oracle_result["summary"],
                    "last_context": context_feats,
                    "known_slots": known_slots,
                    "last_topic": oracle_result["topic"],
                },
                merge=True
            )

            line_bot_api.reply_message(reply_token, TextSendMessage(text=f"{bridge}\n\n{reply_text}"))
            return

        if missing and should_offer_quick_reply(user_data, missing):
            slot = missing[0]
            qr = build_quick_reply_for_slot(slot, topic)
            prompt = build_slot_prompt(slot, topic)
            user_ref.set(
                {
                    "last_question": prompt,
                    "missing_slots": missing,
                    "ui_state": ui_state_name_for_slot(slot),
                    "last_quick_reply_context": {
                        "slot": slot,
                        "topic": topic,
                        "asked_at": now_utc().isoformat(),
                    },
                },
                merge=True
            )
            intro = "その迷い、たしかに受け取りました。まだ答えを急がず、少しだけ流れの輪郭を確かめさせてください。"
            line_bot_api.reply_message(reply_token, TextSendMessage(text=f"{intro}\n{prompt}", quick_reply=qr))
            return

        question_map = {
            "time_continuity": "その悩み、最近急に強くなった感じですか？ それとも前から続いていましたか？",
            "emotion": "今の気持ちにいちばん近いのはどれですか？ 不安 / 悲しさ / イライラ / 空虚さ",
            "desired_action": "今は動きたいですか？ それとも少し様子を見たい気持ちの方が近いですか？",
            "relationship_distance": "相手との距離は今どうですか？ 近い感じですか、それとも少し離れていますか？",
            "main_stressor": "いちばんしんどいのはどれに近いですか？ 仕事量 / 人間関係 / 将来の不安",
            "person_type": "その相手は誰に近いですか？ 家族 / 友人 / 職場 / 恋人"
        }
        next_question = question_map.get(missing[0], "もう少しだけ、そのまま話してください。")

        user_ref.set({"last_question": next_question, "missing_slots": missing}, merge=True)
        intro = "その迷い、たしかに受け取りました。まだ答えを急がず、少しだけ流れの輪郭を確かめさせてください。"
        line_bot_api.reply_message(reply_token, TextSendMessage(text=f"{intro}\n{next_question}"))
        return

    if route == "continue_consult":
        current_topic = user_data.get("current_topic") or oracle_engine.topic_classifier.classify(user_text)
        active_text = user_data.get("active_consultation_text") or user_data.get("last_consultation_text") or user_text
        known_slots = user_data.get("known_slots") or {}
        extracted = extract_slots_from_text(user_text, current_topic)
        known_slots = merge_known_slots(known_slots, extracted)

        user_ref.set(
            {
                "conversation_mode": "consulting",
                "current_topic": current_topic,
                "active_consultation_text": active_text,
                "known_slots": known_slots,
                "ui_state": "none",
            },
            merge=True
        )

        needed = required_slots_for_topic(current_topic, user_data.get("plan_tier", "free"))
        filled = [k for k in needed if known_slots.get(k)]
        missing = [s for s in needed if not known_slots.get(s)]

        if len(filled) >= min(3, len(needed)):
            bridge = "だいぶ輪郭が見えてきました。では、今のあなたに近い流れを言葉にします。"
            reply_text, oracle_result, context_feats = build_reading_reply(
                user_data={**user_data, "current_topic": current_topic},
                active_text=active_text,
                known_slots=known_slots
            )

            if not user_data.get("birth_date"):
                birth_qr = build_birthdate_quick_reply()
                user_ref.set(
                    {
                        "conversation_mode": "post_oracle",
                        "known_slots": known_slots,
                        "missing_slots": [],
                        "last_question": firestore.DELETE_FIELD,
                        "last_oracle_message": oracle_result["message"],
                        "last_oracle_summary": oracle_result["summary"],
                        "last_context": context_feats,
                        "last_topic": oracle_result["topic"],
                        "ui_state": "awaiting_birthdate",
                    },
                    merge=True
                )
                line_bot_api.reply_message(reply_token, TextSendMessage(text=f"{bridge}\n\n{reply_text}", quick_reply=birth_qr))
                return

            user_ref.set(
                {
                    "conversation_mode": "post_oracle",
                    "known_slots": known_slots,
                    "missing_slots": [],
                    "last_question": firestore.DELETE_FIELD,
                    "last_oracle_message": oracle_result["message"],
                    "last_oracle_summary": oracle_result["summary"],
                    "last_context": context_feats,
                    "last_topic": oracle_result["topic"],
                },
                merge=True
            )
            line_bot_api.reply_message(reply_token, TextSendMessage(text=f"{bridge}\n\n{reply_text}"))
            return

        if missing and should_offer_quick_reply(user_data, missing):
            slot = missing[0]
            qr = build_quick_reply_for_slot(slot, current_topic)
            prompt = build_slot_prompt(slot, current_topic)
            user_ref.set(
                {
                    "last_question": prompt,
                    "missing_slots": missing,
                    "ui_state": ui_state_name_for_slot(slot),
                    "last_quick_reply_context": {
                        "slot": slot,
                        "topic": current_topic,
                        "asked_at": now_utc().isoformat(),
                    },
                },
                merge=True
            )
            bridge = "少しずつ輪郭が見えてきました。"
            line_bot_api.reply_message(reply_token, TextSendMessage(text=f"{bridge}\n{prompt}", quick_reply=qr))
            return

        question_map = {
            "time_continuity": "その悩みって、急に強くなった感じですか？ それとも前からずっと続いていましたか？",
            "emotion": "今の気持ちにいちばん近いのはどれですか？ 不安 / 悲しさ / イライラ / 空虚さ",
            "desired_action": "今は動きたいですか？ それとも少し様子を見たい気持ちの方が近いですか？",
            "relationship_distance": "相手との距離は今どうですか？ 近い感じですか、それとも少し離れていますか？",
            "main_stressor": "いちばんしんどいのはどれに近いですか？ 仕事量 / 人間関係 / 将来の不安",
            "person_type": "その相手は誰に近いですか？ 家族 / 友人 / 職場 / 恋人"
        }
        next_question = question_map.get(missing[0], "もう少しだけ、そのまま話してください。")

        user_ref.set({"last_question": next_question, "missing_slots": missing}, merge=True)
        bridge = "少しずつ輪郭が見えてきました。"
        line_bot_api.reply_message(reply_token, TextSendMessage(text=f"{bridge}\n{next_question}"))
        return

    line_bot_api.reply_message(reply_token, TextSendMessage(text="そのまま話してください。必要なところだけ、こちらで拾っていきます。"))


# -------------------------
# Postback helpers
# -------------------------
def parse_postback_data(data: str) -> dict:
    result = {}
    if not data:
        return result
    for part in data.split("&"):
        if "=" in part:
            k, v = part.split("=", 1)
            result[k] = v
    return result


def apply_quick_reply_selection(user_data: dict, payload: dict) -> dict:
    known_slots = dict(user_data.get("known_slots") or {})
    slot = payload.get("slot")
    value = payload.get("value")

    if slot and value:
        known_slots[slot] = value

    return known_slots


def handle_postback_flow(reply_token: str, user_ref, user_data: dict, event: PostbackEvent):
    data = parse_postback_data(event.postback.data)

    # birthdate picker
    if data.get("type") == "birthdate_picker":
        picked_date = getattr(event.postback, "params", {}).get("date")
        if not picked_date:
            line_bot_api.reply_message(
                reply_token,
                TextSendMessage(text="日付の受け取りに少し乱れがありました。テキストで送っても大丈夫です。")
            )
            return

        user_ref.set({"birth_date": picked_date, "ui_state": "none", "pending_birthdate_request": False}, merge=True)

        active_text = user_data.get("active_consultation_text") or user_data.get("last_consultation_text")
        current_topic = user_data.get("current_topic") or user_data.get("last_topic")
        known_slots = user_data.get("known_slots") or {}

        if active_text and current_topic:
            refreshed_user_data = {**user_data, "birth_date": picked_date}
            reply_text, oracle_result, context_feats = build_reading_reply(
                user_data=refreshed_user_data,
                active_text=active_text,
                known_slots=known_slots
            )
            user_ref.set(
                {
                    "last_oracle_message": oracle_result["message"],
                    "last_oracle_summary": oracle_result["summary"],
                    "last_context": context_feats,
                    "conversation_mode": "post_oracle",
                },
                merge=True
            )
            line_bot_api.reply_message(
                reply_token,
                TextSendMessage(
                    text=(
                        f"生まれた日の気配を受け取りました。{picked_date} として記録しておきます。\n\n"
                        f"その巡りも重ねて、今の流れをもう一度視ました。\n{reply_text}"
                    )
                )
            )
            return

        line_bot_api.reply_message(
            reply_token,
            TextSendMessage(text=f"生まれた日の気配を受け取りました。{picked_date} として記録しておきます。")
        )
        return

    # birthdate manual
    if data.get("type") == "birthdate_manual":
        user_ref.set({"ui_state": "awaiting_birthdate", "pending_birthdate_request": True}, merge=True)
        line_bot_api.reply_message(
            reply_token,
            TextSendMessage(text="生年月日をそのまま送ってください。西暦でも和暦でも大丈夫です。")
        )
        return

    # birthdate skip
    if data.get("type") == "birthdate_skip":
        user_ref.set({"ui_state": "none", "pending_birthdate_request": False}, merge=True)
        line_bot_api.reply_message(
            reply_token,
            TextSendMessage(text="わかりました。必要になったら、あとからでも預けられます。")
        )
        return

    # escape hatch / slot select
    if "slot" in data and "value" in data:
        slot = data["slot"]
        value = data["value"]

        if value == "__UNRESOLVED__":
            user_ref.set(
                {
                    "ui_state": "none",
                    "last_quick_reply_context": firestore.DELETE_FIELD,
                    "conversation_mode": "consulting",
                },
                merge=True
            )
            line_bot_api.reply_message(reply_token, TextSendMessage(text=unresolved_slot_response(slot)))
            return

        known_slots = apply_quick_reply_selection(user_data, data)
        current_topic = user_data.get("current_topic") or user_data.get("last_topic") or "relationship"
        active_text = user_data.get("active_consultation_text") or user_data.get("last_consultation_text") or ""
        plan_tier = user_data.get("plan_tier", "free")
        needed = required_slots_for_topic(current_topic, plan_tier)
        filled = [k for k in needed if known_slots.get(k)]
        missing = [s for s in needed if not known_slots.get(s)]

        user_ref.set(
            {
                "known_slots": known_slots,
                "ui_state": "none",
                "last_quick_reply_context": firestore.DELETE_FIELD,
                "conversation_mode": "consulting",
            },
            merge=True
        )

        if len(filled) >= min(3, len(needed)):
            bridge = "だいぶ輪郭が見えてきました。では、今のあなたに近い流れを言葉にします。"
            reply_text, oracle_result, context_feats = build_reading_reply(
                user_data={**user_data, "current_topic": current_topic},
                active_text=active_text or current_topic,
                known_slots=known_slots
            )

            if not user_data.get("birth_date"):
                birth_qr = build_birthdate_quick_reply()
                user_ref.set(
                    {
                        "conversation_mode": "post_oracle",
                        "last_oracle_message": oracle_result["message"],
                        "last_oracle_summary": oracle_result["summary"],
                        "last_context": context_feats,
                        "last_topic": oracle_result["topic"],
                        "ui_state": "awaiting_birthdate",
                    },
                    merge=True
                )
                line_bot_api.reply_message(reply_token, TextSendMessage(text=f"{bridge}\n\n{reply_text}", quick_reply=birth_qr))
                return

            user_ref.set(
                {
                    "conversation_mode": "post_oracle",
                    "last_oracle_message": oracle_result["message"],
                    "last_oracle_summary": oracle_result["summary"],
                    "last_context": context_feats,
                    "last_topic": oracle_result["topic"],
                },
                merge=True
            )
            line_bot_api.reply_message(reply_token, TextSendMessage(text=f"{bridge}\n\n{reply_text}"))
            return

        if missing and should_offer_quick_reply(user_data, missing):
            next_slot = missing[0]
            qr = build_quick_reply_for_slot(next_slot, current_topic)
            prompt = build_slot_prompt(next_slot, current_topic)
            user_ref.set(
                {
                    "last_question": prompt,
                    "missing_slots": missing,
                    "ui_state": ui_state_name_for_slot(next_slot),
                    "last_quick_reply_context": {
                        "slot": next_slot,
                        "topic": current_topic,
                        "asked_at": now_utc().isoformat(),
                    },
                },
                merge=True
            )
            line_bot_api.reply_message(reply_token, TextSendMessage(text=prompt, quick_reply=qr))
            return

        line_bot_api.reply_message(reply_token, TextSendMessage(text="そのまま続けて話してください。必要な輪郭は少しずつ見えてきています。"))
        return

    line_bot_api.reply_message(reply_token, TextSendMessage(text="そのまま続けてください。必要なところだけ、こちらで拾っていきます。"))


# -------------------------
# API
# -------------------------
@app.get("/")
def root():
    return {"status": "online", "message": "SHIKI System is running."}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "firebase_initialized": bool(firebase_admin._apps),
        "gemini_model": GEMINI_MODEL,
        "timestamp": now_utc().isoformat(),
    }


@app.get("/morning-push")
def morning_push():
    try:
        users_ref = db.collection("users").stream()
        count = 0

        for user in users_ref:
            u_id = user.id
            u_data = user.to_dict() or {}
            last_msg = u_data.get("last_msg", "静かな心")
            prompt = (
                "あなたは神秘的な存在『識（SHIKI）』です。"
                f"ユーザーの昨日の言葉: {last_msg}\n"
                "今日を歩むための短い一言を80文字以内で作り、最後に『――識より』を添えてください。"
            )
            msg_text = safe_generate(prompt, "新しい朝が来ました。そのままのあなたで。――識より")
            line_bot_api.push_message(u_id, TextSendMessage(text=msg_text))
            count += 1

        return {"status": "completed", "sent_count": count}
    except Exception as e:
        logger.exception("Push error")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature")
    body = await request.body()

    if not signature:
        raise HTTPException(status_code=400, detail="Missing X-Line-Signature header")

    body_text = body.decode("utf-8")

    try:
        handler.handle(body_text, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logger.exception("Webhook handling error")
        raise HTTPException(status_code=500, detail=f"Webhook handling failed: {e}")

    return "OK"


# -------------------------
# Event Dispatcher
# -------------------------
@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event: MessageEvent):
    try:
        u_id = event.source.user_id
        user_text = event.message.text.strip()

        user_ref, user_data = get_or_create_user(u_id)

        # 先に自然消滅チェック（前回 last_active ベース）
        user_data = implicit_suspend_check(user_ref, user_data)

        # その後 touch
        user_data = apply_common_user_touch(user_ref, user_data, last_msg=user_text)

        fire_and_forget(update_listener_snapshot_async, user_ref, user_text)

        handle_message_flow(event.reply_token, user_ref, user_data, user_text)

    except Exception:
        logger.exception("handle_text_message error")
        try:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="識の観測にわずかな乱れが生じました。少し時間を置いて、また同期してください。")
            )
        except Exception:
            logger.exception("Reply fallback failed")


@handler.add(PostbackEvent)
def handle_postback(event: PostbackEvent):
    try:
        u_id = event.source.user_id
        user_ref, user_data = get_or_create_user(u_id)

        user_data = implicit_suspend_check(user_ref, user_data)
        user_data = apply_common_user_touch(user_ref, user_data)

        handle_postback_flow(event.reply_token, user_ref, user_data, event)

    except Exception:
        logger.exception("handle_postback error")
        try:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="識の観測にわずかな乱れが生じました。少し時間を置いて、また同期してください。")
            )
        except Exception:
            logger.exception("Reply fallback failed")
