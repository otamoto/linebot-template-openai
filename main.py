import os
import json
import logging
import re
import threading
import unicodedata
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Dict, Any, List

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

from google import genai

import firebase_admin
from firebase_admin import credentials, firestore

from oracle_engine import EngineState, OracleEngine


# -------------------------
# ログ
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s : %(message)s",
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
genai_client = genai.Client(api_key=GEMINI_API_KEY)


# -------------------------
# Oracle Engine
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
# Persona
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
- 会話修復が必要なときは、情報収集より違和感の受け止めを優先する
- 相槌には重く返しすぎない
""".strip()


# -------------------------
# 基本ユーティリティ
# -------------------------
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


# Firestore互換のため UTC aware datetime を基本にする

def ensure_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(value))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", (text or "").strip())


def parse_japanese_era_date(text: str) -> Optional[str]:
    text = normalize_text(text)
    era_map = {"昭和": 1925, "平成": 1988, "令和": 2018}

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

    patterns = [
        r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$",
        r"^(\d{4})年(\d{1,2})月(\d{1,2})日$",
        r"^(\d{4})(\d{2})(\d{2})$",
        r"^(\d{4})(\d{1,2})(\d{1,2})$",
    ]
    for pat in patterns:
        m = re.match(pat, text)
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
        "そうか", "ふむ", "んー", "うーむ", "へえ", "ほう", "はい", "うん",
        "わかった", "ありがとう"
    }
    return len(t) <= 8 and t in short_set


def is_new_consult_message(text: str) -> bool:
    t = normalize_text(text)
    keywords = [
        "それとは別に", "ちなみに", "別件", "他にも", "仕事のことも",
        "家族のことも", "別の相談", "新しく", "別の話"
    ]
    return any(k in t for k in keywords)


def is_hard_reset_message(text: str) -> bool:
    t = normalize_text(text)
    keywords = [
        "最初から", "リセット", "新しく始めたい", "最初からいい", "最初からお願い",
        "一回リセット", "やり直したい", "新しく相談したい", "仕切り直したい"
    ]
    return any(k in t for k in keywords)


def is_resume_like_message(text: str) -> bool:
    t = normalize_text(text)
    keywords = ["続き", "その後", "前回", "あれから", "変化", "まだ", "やっぱり"]
    return any(k in t for k in keywords)


def looks_like_full_consult_message(text: str) -> bool:
    t = normalize_text(text)
    if len(t) < 10:
        return False
    consult_markers = [
        "迷っています", "悩んでいます", "どうしたら", "どうすれば", "つらい",
        "苦しい", "好きな人", "仕事", "上司", "家族", "友達", "人間関係",
        "連絡するべき", "別れるべき", "転職", "不安", "悩み"
    ]
    return any(k in t for k in consult_markers)


def append_consultation_text(existing: str, new_text: str, max_chars: int = 900) -> str:
    existing = (existing or "").strip()
    new_text = (new_text or "").strip()

    if not existing:
        combined = new_text
    else:
        combined = existing + "\n" + new_text

    if len(combined) > max_chars:
        combined = combined[-max_chars:]
    return combined


def required_slots_for_topic(topic: str, plan_tier: str) -> list:
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
        "patience": float(user_data.get("patience", 0.45)),
    }


def build_memory(user_data: dict) -> dict:
    return {
        "repeat_count": int(user_data.get("repeat_count", 1)),
        "volatility": float(user_data.get("volatility", 0.55)),
    }


def build_base_context(user_data: dict) -> dict:
    return {
        "stress": float(user_data.get("stress", 0.60)),
        "sleep_deficit": float(user_data.get("sleep_deficit", 0.50)),
        "loneliness": float(user_data.get("loneliness", 0.55)),
        "urgency": float(user_data.get("urgency", 0.65)),
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


def build_oracle_input_text(active_text: str, known_slots: dict) -> str:
    lines = [(active_text or "").strip()]
    for k, v in (known_slots or {}).items():
        if v:
            lines.append(f"{k}: {v}")
    return "\n".join([x for x in lines if x])


def json_from_text(text: str) -> Optional[dict]:
    if not text:
        return None

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    m = re.search(r"\{.*\}", cleaned, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def gemini_text(prompt: str, fallback: str) -> str:
    try:
        response = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        text = getattr(response, "text", "") or ""
        text = text.strip()
        return text if text else fallback
    except Exception:
        logger.exception("Gemini text generation failed")
        return fallback


def gemini_json(prompt: str, fallback: dict) -> dict:
    try:
        response = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        text = getattr(response, "text", "") or ""
        parsed = json_from_text(text)
        return parsed if isinstance(parsed, dict) else fallback
    except Exception:
        logger.exception("Gemini JSON generation failed")
        return fallback


# -------------------------
# Gemini 補助
# -------------------------
def classify_relation_to_previous_with_gemini(user_text: str, bookmark_summary: str, previous_topic: str) -> dict:
    fallback = {
        "relation_to_previous": "unclear",
        "confidence": 0.40,
        "reason_short": "fallback",
    }

    prompt = f"""
あなたは会話の関係判定器です。JSONのみ返してください。

入力:
- previous_topic: {previous_topic}
- bookmark_summary: {bookmark_summary}
- user_text: {user_text}

relation_to_previous は次のいずれか:
resume
new_consult
ambient
unclear

JSON形式:
{{
  "relation_to_previous": "...",
  "confidence": 0.0,
  "reason_short": "..."
}}
""".strip()

    result = gemini_json(prompt, fallback)
    if result.get("relation_to_previous") not in {"resume", "new_consult", "ambient", "unclear"}:
        return fallback
    return result


def classify_turn_with_gemini(user_text: str, mode: str, bookmark_exists: bool) -> dict:
    fallback = {
        "turn_zone": "consultation" if mode == "consulting" else "post_oracle" if mode == "post_oracle" else "unknown",
        "intent": "continue_consult" if mode == "consulting" else "ambient_response" if mode == "post_oracle" else "unknown",
        "confidence": 0.40,
        "contains_new_consult_content": False,
        "contains_question": False,
        "contains_process_discomfort": False,
        "should_avoid_quick_reply": False,
        "reason_short": "fallback",
    }

    prompt = f"""
あなたはLINE Bot『識（SHIKI）』の会話機能分類器です。
あなたは返答を作りません。JSONのみ返してください。

入力:
- mode: {mode}
- bookmark_exists: {str(bookmark_exists).lower()}
- user_text: {user_text}

まず turn_zone を次から1つ選んでください:
- session_control
- process_repair
- post_oracle
- consultation
- unknown

次に intent を次から1つ選んでください:
- hard_reset
- resume_session
- start_consult
- continue_consult
- new_consult_reset
- clarify_oracle
- ask_action
- deepen_topic
- disagree_oracle
- challenge_bot
- clarify_process
- ambient_response
- unknown

意味の基準:
- challenge_bot: Botそのものへの違和感、不信、テンプレ感、会話になっていない等
- clarify_process: 進め方や質問意図への確認。「これだけで見えるの？」「今なにを聞いてるの？」等
- clarify_oracle: 直前の神託や説明の意味確認
- ask_action: どう動くか、何を避けるか、何をするとよいか
- deepen_topic: もっと深く見てほしい、別角度でさらに見てほしい
- disagree_oracle: 神託内容とのズレ、解釈違い
- ambient_response: 相槌、受け取り、短い返答。情報追加も質問もほぼない
- continue_consult: 状況説明・追加情報・相談本文の継続
- start_consult: 新しく相談が始まる本文

重要ルール:
- Botや進め方への違和感は consultation にしない
- 神託後の短い相槌は ambient_response にする
- 「はい」「なるほど」「わかった」「そうか」は ambient_response が有力
- 情報不足だからといって continue_consult に寄せすぎない
- 曖昧なら unknown を返してよい
- process_repair と ambient_response のときは should_avoid_quick_reply を true にしやすい
- 神託の意味を尋ねているなら clarify_oracle
- 行動を尋ねているなら ask_action
- 相談本文・具体事情・近況追加なら consultation

JSON形式:
{{
  "turn_zone": "...",
  "intent": "...",
  "confidence": 0.0,
  "contains_new_consult_content": true,
  "contains_question": false,
  "contains_process_discomfort": false,
  "should_avoid_quick_reply": false,
  "reason_short": "..."
}}
""".strip()

    result = gemini_json(prompt, fallback)

    valid_turn_zones = {"session_control", "process_repair", "post_oracle", "consultation", "unknown"}
    valid_intents = {
        "hard_reset", "resume_session", "start_consult", "continue_consult", "new_consult_reset",
        "clarify_oracle", "ask_action", "deepen_topic", "disagree_oracle",
        "challenge_bot", "clarify_process", "ambient_response", "unknown"
    }

    if result.get("turn_zone") not in valid_turn_zones:
        result["turn_zone"] = fallback["turn_zone"]
    if result.get("intent") not in valid_intents:
        result["intent"] = fallback["intent"]

    result["contains_new_consult_content"] = bool(result.get("contains_new_consult_content", False))
    result["contains_question"] = bool(result.get("contains_question", False))
    result["contains_process_discomfort"] = bool(result.get("contains_process_discomfort", False))
    result["should_avoid_quick_reply"] = bool(result.get("should_avoid_quick_reply", False))
    return result


def extract_slots_with_gemini(user_text: str, topic: str) -> dict:
    fallback = {}

    prompt = f"""
あなたは相談文からスロット情報を抽出するJSON関数です。JSONのみ返してください。

入力:
- topic: {topic}
- user_text: {user_text}

使えるスロット:
共通:
- time_continuity: "急に強くなった" / "前から続いている" / "__UNRESOLVED__"
- emotion: "不安が強い" / "悲しさや寂しさが強い" / "怒りや苛立ちが強い" / "空虚さが強い" / "__UNRESOLVED__"
- desired_action: "動きたい" / "待ちたい" / "終わらせたい" / "__UNRESOLVED__"

love:
- relationship_distance: "近い" / "少し離れている" / "かなり離れている"

work:
- main_stressor: "仕事量" / "人間関係" / "将来の不安"

relationship:
- person_type: "家族" / "友人" / "職場" / "恋人"

抽出できないものは省略してください。
JSONのみ返してください。
""".strip()

    result = gemini_json(prompt, fallback)
    return result if isinstance(result, dict) else {}


# -------------------------
# Quick Reply builders
# -------------------------
def make_quick_reply(items: list) -> QuickReply:
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
                display_text="自分で入力する",
            )
        ),
        QuickReplyButton(
            action=PostbackAction(
                label="今は送らない",
                data="type=birthdate_skip",
                display_text="今は送らない",
            )
        ),
    ]
    return make_quick_reply(items)


def build_relation_confirm_quick_reply() -> QuickReply:
    items = [
        QuickReplyButton(action=PostbackAction(label="前回の続き", data="type=relation_confirm&value=resume", display_text="前回の続き")),
        QuickReplyButton(action=PostbackAction(label="別の相談", data="type=relation_confirm&value=new", display_text="別の相談")),
        QuickReplyButton(action=PostbackAction(label="まだうまく言えない", data="type=relation_confirm&value=unclear", display_text="まだうまく言えない")),
    ]
    return make_quick_reply(items)


def should_offer_quick_reply(user_data: dict, missing_slots: list, turn_meta: Optional[dict] = None) -> bool:
    if not missing_slots:
        return False
    if user_data.get("ui_state", "none") != "none":
        return False

    turn_meta = turn_meta or {}
    if turn_meta.get("should_avoid_quick_reply"):
        return False

    if turn_meta.get("turn_zone") in {"process_repair", "post_oracle"}:
        return False

    if turn_meta.get("intent") in {
        "ambient_response",
        "challenge_bot",
        "clarify_process",
        "clarify_oracle",
        "ask_action",
        "deepen_topic",
        "disagree_oracle",
        "unknown",
    }:
        return False

    last_ctx = user_data.get("last_quick_reply_context") or {}
    if last_ctx.get("slot") == missing_slots[0]:
        return False

    recent_qr_count = int(user_data.get("recent_quick_reply_count", 0))
    if recent_qr_count >= 1:
        return False

    if user_data.get("last_user_ignored_quick_reply", False):
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
# 文体バリエーション / 定型文抑制
# -------------------------
RESPONSE_VARIANTS = {
    "consult_intro": [
        "まだ断定の段階ではありません。先に揺れの濃いところだけ拾わせてください。",
        "いまは結論より、濃く動いている部分だけを拾います。",
        "まずは盤面の濃い部分を静かに見ていきます。",
    ],
    "reading_bridge": [
        "今の揺れ方はある程度見えてきました。では、この流れを言葉にします。",
        "輪郭が少し見えてきました。今の流れをそのまま言葉にします。",
        "いま見えている流れを、静かに言葉へ移します。",
    ],
    "continue_prompt": [
        "もう少しだけ、そのまま話してください。",
        "続けられそうなら、もう少しだけ聞かせてください。",
        "あと少しだけ、いま引っかかっているところを話してみてください。",
    ],
    "deepen_intro": [
        "では、さっきの流れをもう少し深く見ます。",
        "では、この揺れの奥を少しだけ見ていきます。",
        "では、今の流れをもう一段だけ深く辿ります。",
    ],
}


def choose_variant(key: str, user_data: dict) -> str:
    options = RESPONSE_VARIANTS.get(key, ["そのまま続けてください。"])
    last_key = user_data.get("last_response_variant_key")
    last_index = int(user_data.get("last_response_variant_index", -1))

    if key == last_key and len(options) > 1:
        next_index = (last_index + 1) % len(options)
    else:
        next_index = 0
    return options[next_index]


def save_variant_state(user_ref, key: str, user_data: dict):
    options = RESPONSE_VARIANTS.get(key, [""])
    last_key = user_data.get("last_response_variant_key")
    last_index = int(user_data.get("last_response_variant_index", -1))

    if key == last_key and len(options) > 1:
        next_index = (last_index + 1) % len(options)
    else:
        next_index = 0

    user_ref.set(
        {
            "last_response_variant_key": key,
            "last_response_variant_index": next_index,
        },
        merge=True,
    )


def remember_sent_phrase(user_ref, user_data: dict, text: str):
    phrases = list(user_data.get("recent_sent_phrases") or [])
    head = (text or "").strip().split("\n")[0][:60]
    if head:
        phrases.append(head)
    phrases = phrases[-5:]
    user_ref.set({"recent_sent_phrases": phrases}, merge=True)


# -------------------------
# 状態管理
# -------------------------
def build_bookmark_payload(user_data: dict) -> dict:
    last_oracle_summary = user_data.get("last_oracle_summary") or {}
    current_topic = user_data.get("current_topic") or user_data.get("last_topic") or "relationship"
    known_slots = user_data.get("known_slots") or {}

    if current_topic == "love":
        next_point = known_slots.get("relationship_distance", "相手との距離感がどう動くか")
    elif current_topic == "work":
        next_point = known_slots.get("main_stressor", "仕事の重さがどう変わるか")
    else:
        next_point = known_slots.get("person_type", "相手との距離感がどう動くか")

    oracle_summary = last_oracle_summary.get("core_meaning", "今は流れを急がず見直す時期")
    core_issue = last_oracle_summary.get("risk_hint", "判断を急ぎやすいこと")
    resume_prompt = (
        f"……再同期します。前回、あなたは『{oracle_summary}』という流れの中にいました。"
        f"その後、{next_point}に変化はありましたか。"
    )

    return {
        "oracle_summary": oracle_summary,
        "core_issue": core_issue,
        "next_observation_point": next_point,
        "resume_prompt": resume_prompt,
        "session_closed_at": now_utc().isoformat(),
    }


def implicit_suspend_check(user_ref, user_data: dict) -> dict:
    mode = user_data.get("conversation_mode", "idle")
    if mode not in {"consulting", "post_oracle", "repairing"}:
        return user_data

    last_active = ensure_datetime(user_data.get("last_active"))
    if not last_active:
        return user_data

    if now_utc() - last_active < timedelta(hours=24):
        return user_data

    bookmark = user_data.get("bookmark")
    if not bookmark:
        bookmark = build_bookmark_payload(user_data)

    patch = {
        "conversation_mode": "suspended",
        "bookmark": bookmark,
        "session_closed_at": now_utc(),
    }
    user_ref.set(patch, merge=True)

    merged = dict(user_data)
    merged.update(patch)
    return merged


def reset_consultation_state(user_ref):
    user_ref.set(
        {
            "conversation_mode": "idle",
            "current_topic": firestore.DELETE_FIELD,
            "active_consultation_text": firestore.DELETE_FIELD,
            "known_slots": {},
            "missing_slots": [],
            "last_question": firestore.DELETE_FIELD,
            "ui_state": "none",
            "repair_pending": False,
            "pending_relation_choice": firestore.DELETE_FIELD,
            "recent_quick_reply_count": 0,
            "last_user_ignored_quick_reply": False,
        },
        merge=True,
    )


def get_or_create_user(user_id: str):
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


def set_mode(user_ref, user_data: dict, mode: str):
    user_ref.set({"conversation_mode": mode}, merge=True)
    user_data["conversation_mode"] = mode


def bump_quick_reply_counter(user_ref, user_data: dict, value: int):
    user_ref.set({"recent_quick_reply_count": value}, merge=True)
    user_data["recent_quick_reply_count"] = value


def mark_ignored_quick_reply(user_ref, user_data: dict, ignored: bool):
    user_ref.set({"last_user_ignored_quick_reply": ignored}, merge=True)
    user_data["last_user_ignored_quick_reply"] = ignored


# -------------------------
# 背景処理
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
        "listener_snapshot_summary": f"latest_text_len={len(t)}",
    }
    user_ref.set({"listener_snapshot": summary}, merge=True)


# -------------------------
# 応答生成
# -------------------------
def build_reading_reply(user_data: dict, active_text: str, known_slots: dict) -> Tuple[str, dict, dict]:
    plan_tier = user_data.get("plan_tier", "free")
    is_paid = plan_tier in {"paid", "deep"}
    horizon = "week" if is_paid else "today"

    base_context = user_data.get("last_context") or build_base_context(user_data)
    context_feats = slots_to_context(base_context, known_slots)
    user_profile = build_user_profile(user_data)
    memory = build_memory(user_data)

    oracle_input_text = build_oracle_input_text(active_text, known_slots)
    oracle_result = oracle_engine.predict(
        user_profile=user_profile,
        context_feats=context_feats,
        user_text=oracle_input_text,
        horizon=horizon,
        memory=memory,
        is_paid=is_paid,
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
        f"簡単に言うと、{oracle_summary.get('core_meaning', '今は流れを急がず見直した方がいい時期です。')}\n"
        f"{oracle_summary.get('risk_hint', '')}"
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

    return gemini_text(prompt, fallback)


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

    return gemini_text(prompt, fallback)


def respond_to_process_challenge(user_text: str, route: str = "clarify_process") -> str:
    if route == "challenge_bot":
        fallback = "そう感じたのはもっともです。まだ輪郭が見えたというより、入口に触れた程度です。急ぎすぎたなら少し戻して、あなたの言葉から丁寧に拾い直します。"
    else:
        fallback = "いまは答えを決めるというより、揺れている場所を確かめている段階です。急ぎすぎるように見えたなら少し戻します。必要なら、今どこを見ているかも短く言葉にします。"

    prompt = f"""
{PERSONA_GUARDRAIL}

ユーザーは、識の進め方や発言に違和感や疑問を示しています。
防御的にならず、自然に認め、進め方を少し整えてください。
2〜4文で返してください。
このターンでは質問攻めにしないでください。
route: {route}

ユーザー発話:
{user_text}
""".strip()

    return gemini_text(prompt, fallback)


def handle_disagreement_repair() -> str:
    return "ズレを感じたなら、それは大事な反応です。どの部分がいちばん違うと感じたのか、そこだけ教えてください。"


def ambient_response(text: str) -> str:
    t = normalize_text(text)

    if t in {"はい", "うん", "なるほど", "そうなんだ", "そうか", "わかった", "ありがとう"}:
        variants = [
            "うん、その受け取り方で大丈夫です。",
            "それだけ受け取れたなら、今は十分です。",
            "いまは、その感触だけ置いておけば大丈夫です。",
            "また続きを見たくなったら、その時に聞かせてください。",
        ]
        return variants[hash(t) % len(variants)]

    if t in {"ん？", "え？", "うーん", "えっと", "ふむ", "へえ", "ほう"}:
        variants = [
            "急がなくて大丈夫です。",
            "まだ言葉にならなくても大丈夫です。",
            "引っかかるところだけでも十分です。",
        ]
        return variants[hash(t) % len(variants)]

    return "そのまま話して大丈夫です。必要なところだけ拾います。"


def build_resume_reply(bookmark: dict) -> str:
    return bookmark.get("resume_prompt", "……再同期します。前回の流れの続きを、もう一度見ていきましょう。")


def unresolved_slot_response(slot_name: str) -> str:
    mapping = {
        "emotion": "無理に形にしなくても大丈夫です。濁ったまま、もう少し奥を見てみましょう。",
        "desired_action": "まだ決めきれなくても大丈夫です。その揺れごと観測していきます。",
        "time_continuity": "うまく分けられなくても大丈夫です。その曖昧さも手がかりになります。",
    }
    return mapping.get(slot_name, "はっきり分けられなくても大丈夫です。そのまま続けていきましょう。")


# -------------------------
# ルーティング
# -------------------------
def route_user_message(user_text: str, user_data: dict) -> Tuple[str, dict]:
    mode = user_data.get("conversation_mode", "idle")

    fallback_turn = {
        "turn_zone": "unknown",
        "intent": "unknown",
        "confidence": 0.0,
        "contains_new_consult_content": False,
        "contains_question": False,
        "contains_process_discomfort": False,
        "should_avoid_quick_reply": False,
        "reason_short": "local_fallback",
    }

    if looks_like_birth_date(user_text):
        return "birthdate_update", fallback_turn

    if is_hard_reset_message(user_text):
        return "hard_reset", fallback_turn

    if mode == "suspended":
        bookmark = user_data.get("bookmark") or {}
        relation = classify_relation_to_previous_with_gemini(
            user_text=user_text,
            bookmark_summary=bookmark.get("oracle_summary", ""),
            previous_topic=user_data.get("last_topic", ""),
        ).get("relation_to_previous", "unclear")

        if relation == "resume":
            return "resume_session", fallback_turn
        if relation == "new_consult":
            return "start_consult", fallback_turn
        if relation == "ambient":
            return "ambient_response", fallback_turn
        return "confirm_relation", fallback_turn

    turn = classify_turn_with_gemini(
        user_text=user_text,
        mode=mode,
        bookmark_exists=bool(user_data.get("bookmark")),
    )
    intent = turn.get("intent", "unknown")
    zone = turn.get("turn_zone", "unknown")
    has_new_consult = turn.get("contains_new_consult_content", False)

    if zone == "process_repair" or intent in {"challenge_bot", "clarify_process"}:
        return (intent if intent in {"challenge_bot", "clarify_process"} else "clarify_process"), turn

    if is_new_consult_message(user_text):
        return "new_consult_reset", turn

    if mode == "post_oracle":
        if intent in {
            "clarify_oracle",
            "ask_action",
            "deepen_topic",
            "disagree_oracle",
            "ambient_response",
        }:
            return intent, turn

        if has_new_consult and intent in {"continue_consult", "start_consult"}:
            return "continue_consult", turn

        if looks_like_full_consult_message(user_text) and has_new_consult:
            return "continue_consult", turn

        return "ambient_response", turn

    if mode in {"consulting", "repairing"}:
        if intent in {"clarify_oracle", "ask_action"}:
            return intent, turn
        if intent in {"continue_consult", "start_consult"}:
            return "continue_consult", turn
        if intent == "ambient_response":
            return "ambient_response", turn
        if intent == "deepen_topic":
            return "continue_consult", turn
        return "continue_consult", turn

    if is_short_ambiguous_utterance(user_text):
        return "ambient_response", turn

    if is_resume_like_message(user_text) and user_data.get("bookmark"):
        return "confirm_relation", turn

    if intent in {"start_consult", "continue_consult"}:
        return "start_consult", turn

    return "start_consult", turn


# -------------------------
# 返信ユーティリティ
# -------------------------
def reply_text(reply_token: str, text: str, quick_reply: Optional[QuickReply] = None):
    line_bot_api.reply_message(reply_token, TextSendMessage(text=text, quick_reply=quick_reply))


def reply_and_remember(reply_token: str, user_ref, user_data: dict, text: str, quick_reply: Optional[QuickReply] = None):
    reply_text(reply_token, text, quick_reply=quick_reply)
    remember_sent_phrase(user_ref, user_data, text)


# -------------------------
# Message flow
# -------------------------
def handle_message_flow(reply_token: str, user_ref, user_data: dict, user_text: str):
    # Quick Reply 中に自由入力が来たら ui_state は外し、無視した履歴を残す
    if user_data.get("ui_state", "none") != "none":
        user_ref.set(
            {
                "ui_state": "none",
                "last_user_ignored_quick_reply": True,
                "recent_quick_reply_count": 0,
            },
            merge=True,
        )
        user_data["ui_state"] = "none"
        user_data["last_user_ignored_quick_reply"] = True
        user_data["recent_quick_reply_count"] = 0
    else:
        mark_ignored_quick_reply(user_ref, user_data, False)

    route, turn_meta = route_user_message(user_text, user_data)
    mode = user_data.get("conversation_mode", "idle")

    if route == "hard_reset":
        reset_consultation_state(user_ref)
        msg = "盤面をいったん静かに戻しました。新しく視たいことを、そのまま話してください。"
        reply_and_remember(reply_token, user_ref, user_data, msg)
        return

    if route == "confirm_relation":
        user_ref.set(
            {
                "ui_state": "awaiting_relation_confirm",
                "pending_relation_choice": {
                    "incoming_text": user_text,
                    "asked_at": now_utc().isoformat(),
                },
                "recent_quick_reply_count": 1,
            },
            merge=True,
        )
        msg = (
            "前に観測した揺れの続きを見てもよさそうですか。"
            "\nそれとも、今回は別の盤面として受け取りましょうか。"
        )
        reply_and_remember(reply_token, user_ref, user_data, msg, quick_reply=build_relation_confirm_quick_reply())
        return

    if route == "birthdate_update":
        parsed_birth = parse_birth_date(user_text)
        if not parsed_birth:
            msg = "その日付はまだうまく読み取れませんでした。西暦や和暦の形で、もう一度送ってみてください。"
            reply_and_remember(reply_token, user_ref, user_data, msg)
            return

        user_ref.set({"birth_date": parsed_birth, "ui_state": "none"}, merge=True)

        active_text = user_data.get("active_consultation_text") or user_data.get("last_consultation_text")
        current_topic = user_data.get("current_topic") or user_data.get("last_topic")
        known_slots = user_data.get("known_slots") or {}

        if active_text and current_topic:
            refreshed_user_data = {**user_data, "birth_date": parsed_birth}
            reply_body, oracle_result, context_feats = build_reading_reply(
                user_data=refreshed_user_data,
                active_text=active_text,
                known_slots=known_slots,
            )
            user_ref.set(
                {
                    "last_oracle_message": oracle_result["message"],
                    "last_oracle_summary": oracle_result["summary"],
                    "last_context": context_feats,
                    "conversation_mode": "post_oracle",
                    "recent_quick_reply_count": 0,
                },
                merge=True,
            )
            msg = (
                f"生まれた日の気配を受け取りました。{parsed_birth} として記録しておきます。\n\n"
                f"さっきの流れを、その巡りも重ねてもう一度視ました。\n{reply_body}"
            )
            reply_and_remember(reply_token, user_ref, user_data, msg)
            return

        msg = f"生まれた日の気配を受け取りました。{parsed_birth} として記録しておきます。"
        reply_and_remember(reply_token, user_ref, user_data, msg)
        return

    if route == "resume_session":
        bookmark = user_data.get("bookmark") or {}
        user_ref.set({"conversation_mode": "consulting", "ui_state": "none", "recent_quick_reply_count": 0}, merge=True)
        msg = build_resume_reply(bookmark)
        reply_and_remember(reply_token, user_ref, user_data, msg)
        return

    if route == "new_consult_reset":
        user_ref.set(
            {
                "conversation_mode": "idle",
                "current_topic": firestore.DELETE_FIELD,
                "active_consultation_text": firestore.DELETE_FIELD,
                "known_slots": {},
                "missing_slots": [],
                "last_question": firestore.DELETE_FIELD,
                "ui_state": "none",
                "repair_pending": False,
                "pending_relation_choice": firestore.DELETE_FIELD,
                "recent_quick_reply_count": 0,
                "last_user_ignored_quick_reply": False,
            },
            merge=True,
        )

        if looks_like_full_consult_message(user_text):
            user_data = {
                **user_data,
                "conversation_mode": "idle",
                "current_topic": None,
                "active_consultation_text": None,
                "known_slots": {},
                "missing_slots": [],
                "last_question": None,
                "ui_state": "none",
                "repair_pending": False,
                "recent_quick_reply_count": 0,
                "last_user_ignored_quick_reply": False,
            }
            route = "start_consult"
        else:
            msg = "盤面を切り替えました。新しく視たいことを、そのまま話してください。"
            reply_and_remember(reply_token, user_ref, user_data, msg)
            return

    if route in {"challenge_bot", "clarify_process"}:
        reply_body = respond_to_process_challenge(user_text, route=route)
        user_ref.set(
            {
                "conversation_mode": "repairing",
                "repair_pending": False,
                "ui_state": "none",
                "recent_quick_reply_count": 0,
            },
            merge=True,
        )
        reply_and_remember(reply_token, user_ref, user_data, reply_body)
        return

    if route == "ambient_response":
        msg = ambient_response(user_text)
        user_ref.set({"recent_quick_reply_count": 0}, merge=True)
        reply_and_remember(reply_token, user_ref, user_data, msg)
        return

    last_oracle_message = user_data.get("last_oracle_message", "")
    last_oracle_summary = user_data.get("last_oracle_summary") or {}

    if route == "clarify_oracle":
        reply_body = explain_oracle_simple(user_text, last_oracle_message, last_oracle_summary)
        user_ref.set({"recent_quick_reply_count": 0}, merge=True)
        reply_and_remember(reply_token, user_ref, user_data, reply_body)
        return

    if route == "ask_action":
        reply_body = explain_oracle_action(user_text, last_oracle_message, last_oracle_summary)
        user_ref.set({"recent_quick_reply_count": 0}, merge=True)
        reply_and_remember(reply_token, user_ref, user_data, reply_body)
        return

    if route == "deepen_topic":
        current_topic = user_data.get("current_topic") or user_data.get("last_topic") or "relationship"
        known_slots = user_data.get("known_slots") or {}
        needed = required_slots_for_topic(current_topic, user_data.get("plan_tier", "free"))
        missing = [s for s in needed if not known_slots.get(s)]

        if missing and should_offer_quick_reply(user_data, missing, turn_meta=turn_meta):
            slot = missing[0]
            qr = build_quick_reply_for_slot(slot, current_topic)
            prompt = build_slot_prompt(slot, current_topic)
            intro = choose_variant("deepen_intro", user_data)
            save_variant_state(user_ref, "deepen_intro", user_data)
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
                    "recent_quick_reply_count": 1,
                },
                merge=True,
            )
            reply_and_remember(reply_token, user_ref, user_data, f"{intro}\n{prompt}", quick_reply=qr)
            return

        next_question = choose_variant("continue_prompt", user_data)
        save_variant_state(user_ref, "continue_prompt", user_data)
        intro = choose_variant("deepen_intro", user_data)
        save_variant_state(user_ref, "deepen_intro", user_data)
        user_ref.set({"conversation_mode": "consulting", "last_question": next_question, "ui_state": "none", "recent_quick_reply_count": 0}, merge=True)
        reply_and_remember(reply_token, user_ref, user_data, f"{intro}\n{next_question}")
        return

    if route == "disagree_oracle":
        user_ref.set({"conversation_mode": "repairing", "repair_pending": True, "recent_quick_reply_count": 0}, merge=True)
        msg = handle_disagreement_repair()
        reply_and_remember(reply_token, user_ref, user_data, msg)
        return

    if user_data.get("repair_pending") and mode in {"post_oracle", "repairing"}:
        known_slots = user_data.get("known_slots") or {}
        current_topic = user_data.get("current_topic") or user_data.get("last_topic") or "relationship"

        extracted = extract_slots_with_gemini(user_text, current_topic)
        known_slots = merge_known_slots(known_slots, extracted)

        prev_active = user_data.get("active_consultation_text") or ""
        active_text = append_consultation_text(prev_active, user_text)

        user_ref.set(
            {
                "conversation_mode": "consulting",
                "repair_pending": False,
                "known_slots": known_slots,
                "active_consultation_text": active_text,
                "last_consultation_text": user_text,
                "ui_state": "none",
                "recent_quick_reply_count": 0,
            },
            merge=True,
        )
        user_data = {
            **user_data,
            "conversation_mode": "consulting",
            "repair_pending": False,
            "known_slots": known_slots,
            "active_consultation_text": active_text,
            "last_consultation_text": user_text,
            "recent_quick_reply_count": 0,
        }
        route = "continue_consult"

    if route == "start_consult":
        topic = oracle_engine.topic_classifier.classify(user_text)
        extracted = extract_slots_with_gemini(user_text, topic)
        known_slots = merge_known_slots({}, extracted)

        active_text = append_consultation_text("", user_text)

        user_ref.set(
            {
                "conversation_mode": "consulting",
                "current_topic": topic,
                "active_consultation_text": active_text,
                "last_consultation_text": user_text,
                "known_slots": known_slots,
                "ui_state": "none",
                "recent_quick_reply_count": 0,
            },
            merge=True,
        )

        needed = required_slots_for_topic(topic, user_data.get("plan_tier", "free"))
        filled = [k for k in needed if k in known_slots]
        missing = [s for s in needed if s not in known_slots]

        if len(filled) >= min(3, len(needed)):
            bridge = choose_variant("reading_bridge", user_data)
            save_variant_state(user_ref, "reading_bridge", user_data)
            reply_body, oracle_result, context_feats = build_reading_reply(
                user_data={**user_data, "current_topic": topic},
                active_text=active_text,
                known_slots=known_slots,
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
                        "recent_quick_reply_count": 1,
                    },
                    merge=True,
                )
                reply_and_remember(reply_token, user_ref, user_data, f"{bridge}\n\n{reply_body}", quick_reply=birth_qr)
                return

            user_ref.set(
                {
                    "conversation_mode": "post_oracle",
                    "last_oracle_message": oracle_result["message"],
                    "last_oracle_summary": oracle_result["summary"],
                    "last_context": context_feats,
                    "known_slots": known_slots,
                    "last_topic": oracle_result["topic"],
                    "recent_quick_reply_count": 0,
                },
                merge=True,
            )
            reply_and_remember(reply_token, user_ref, user_data, f"{bridge}\n\n{reply_body}")
            return

        if missing and should_offer_quick_reply(user_data, missing, turn_meta=turn_meta):
            slot = missing[0]
            qr = build_quick_reply_for_slot(slot, topic)
            prompt = build_slot_prompt(slot, topic)
            intro = choose_variant("consult_intro", user_data)
            save_variant_state(user_ref, "consult_intro", user_data)
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
                    "recent_quick_reply_count": 1,
                },
                merge=True,
            )
            reply_and_remember(reply_token, user_ref, user_data, f"{intro}\n{prompt}", quick_reply=qr)
            return

        next_question = choose_variant("continue_prompt", user_data)
        save_variant_state(user_ref, "continue_prompt", user_data)
        intro = choose_variant("consult_intro", user_data)
        save_variant_state(user_ref, "consult_intro", user_data)
        user_ref.set({"last_question": next_question, "missing_slots": missing, "recent_quick_reply_count": 0}, merge=True)
        reply_and_remember(reply_token, user_ref, user_data, f"{intro}\n{next_question}")
        return

    if route == "continue_consult":
        current_topic = user_data.get("current_topic") or oracle_engine.topic_classifier.classify(user_text)

        prev_active = user_data.get("active_consultation_text") or user_data.get("last_consultation_text") or ""
        active_text = append_consultation_text(prev_active, user_text)

        known_slots = user_data.get("known_slots") or {}
        extracted = extract_slots_with_gemini(user_text, current_topic)
        known_slots = merge_known_slots(known_slots, extracted)

        user_ref.set(
            {
                "conversation_mode": "consulting",
                "current_topic": current_topic,
                "active_consultation_text": active_text,
                "last_consultation_text": user_text,
                "known_slots": known_slots,
                "ui_state": "none",
                "recent_quick_reply_count": 0,
            },
            merge=True,
        )

        needed = required_slots_for_topic(current_topic, user_data.get("plan_tier", "free"))
        filled = [k for k in needed if k in known_slots]
        missing = [s for s in needed if s not in known_slots]

        if len(filled) >= min(3, len(needed)):
            bridge = choose_variant("reading_bridge", user_data)
            save_variant_state(user_ref, "reading_bridge", user_data)
            reply_body, oracle_result, context_feats = build_reading_reply(
                user_data={**user_data, "current_topic": current_topic},
                active_text=active_text,
                known_slots=known_slots,
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
                        "recent_quick_reply_count": 1,
                    },
                    merge=True,
                )
                reply_and_remember(reply_token, user_ref, user_data, f"{bridge}\n\n{reply_body}", quick_reply=birth_qr)
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
                    "recent_quick_reply_count": 0,
                },
                merge=True,
            )
            reply_and_remember(reply_token, user_ref, user_data, f"{bridge}\n\n{reply_body}")
            return

        if missing and should_offer_quick_reply(user_data, missing, turn_meta=turn_meta):
            slot = missing[0]
            qr = build_quick_reply_for_slot(slot, current_topic)
            prompt = build_slot_prompt(slot, current_topic)
            intro = choose_variant("consult_intro", user_data)
            save_variant_state(user_ref, "consult_intro", user_data)
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
                    "recent_quick_reply_count": 1,
                },
                merge=True,
            )
            reply_and_remember(reply_token, user_ref, user_data, f"{intro}\n{prompt}", quick_reply=qr)
            return

        next_question = choose_variant("continue_prompt", user_data)
        save_variant_state(user_ref, "continue_prompt", user_data)
        user_ref.set({"last_question": next_question, "missing_slots": missing, "recent_quick_reply_count": 0}, merge=True)
        reply_and_remember(reply_token, user_ref, user_data, next_question)
        return

    msg = "そのまま話して大丈夫です。必要なところだけ、こちらで拾います。"
    reply_and_remember(reply_token, user_ref, user_data, msg)


# -------------------------
# Postback flow
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

    if data.get("type") == "relation_confirm":
        choice = data.get("value")
        pending = user_data.get("pending_relation_choice") or {}
        incoming_text = pending.get("incoming_text", "")

        user_ref.set(
            {
                "ui_state": "none",
                "pending_relation_choice": firestore.DELETE_FIELD,
                "recent_quick_reply_count": 0,
            },
            merge=True,
        )

        if choice == "resume":
            bookmark = user_data.get("bookmark") or {}
            user_ref.set({"conversation_mode": "consulting"}, merge=True)
            msg = build_resume_reply(bookmark)
            reply_and_remember(reply_token, user_ref, user_data, msg)
            return

        if choice == "new":
            reset_consultation_state(user_ref)
            if incoming_text and looks_like_full_consult_message(incoming_text):
                refreshed = {
                    **user_data,
                    "conversation_mode": "idle",
                    "current_topic": None,
                    "known_slots": {},
                    "ui_state": "none",
                    "recent_quick_reply_count": 0,
                }
                handle_message_flow(reply_token, user_ref, refreshed, incoming_text)
                return

            msg = "わかりました。では今回は別の盤面として受け取ります。新しく視たいことを、そのまま話してください。"
            reply_and_remember(reply_token, user_ref, user_data, msg)
            return

        msg = "無理に分けなくても大丈夫です。いま気になっていることを、そのまま少しだけ言葉にしてみてください。"
        reply_and_remember(reply_token, user_ref, user_data, msg)
        return

    if data.get("type") == "birthdate_picker":
        params = getattr(event.postback, "params", None) or {}
        picked_date = params.get("date")
        if not picked_date:
            msg = "日付の受け取りに少し乱れがありました。テキストで送っても大丈夫です。"
            reply_and_remember(reply_token, user_ref, user_data, msg)
            return

        user_ref.set({"birth_date": picked_date, "ui_state": "none", "pending_birthdate_request": False, "recent_quick_reply_count": 0}, merge=True)

        active_text = user_data.get("active_consultation_text") or user_data.get("last_consultation_text")
        current_topic = user_data.get("current_topic") or user_data.get("last_topic")
        known_slots = user_data.get("known_slots") or {}

        if active_text and current_topic:
            refreshed_user_data = {**user_data, "birth_date": picked_date}
            reply_body, oracle_result, context_feats = build_reading_reply(
                user_data=refreshed_user_data,
                active_text=active_text,
                known_slots=known_slots,
            )
            user_ref.set(
                {
                    "last_oracle_message": oracle_result["message"],
                    "last_oracle_summary": oracle_result["summary"],
                    "last_context": context_feats,
                    "conversation_mode": "post_oracle",
                },
                merge=True,
            )
            msg = (
                f"生まれた日の気配を受け取りました。{picked_date} として記録しておきます。\n\n"
                f"その巡りも重ねて、今の流れをもう一度視ました。\n{reply_body}"
            )
            reply_and_remember(reply_token, user_ref, user_data, msg)
            return

        msg = f"生まれた日の気配を受け取りました。{picked_date} として記録しておきます。"
        reply_and_remember(reply_token, user_ref, user_data, msg)
        return

    if data.get("type") == "birthdate_manual":
        user_ref.set({"ui_state": "awaiting_birthdate", "pending_birthdate_request": True, "recent_quick_reply_count": 0}, merge=True)
        msg = "生年月日をそのまま送ってください。西暦でも和暦でも大丈夫です。"
        reply_and_remember(reply_token, user_ref, user_data, msg)
        return

    if data.get("type") == "birthdate_skip":
        user_ref.set({"ui_state": "none", "pending_birthdate_request": False, "recent_quick_reply_count": 0}, merge=True)
        msg = "わかりました。必要になったら、あとからでも預けられます。"
        reply_and_remember(reply_token, user_ref, user_data, msg)
        return

    if "slot" in data and "value" in data:
        slot = data["slot"]
        value = data["value"]

        if value == "__UNRESOLVED__":
            known_slots = dict(user_data.get("known_slots") or {})
            known_slots[slot] = "__UNRESOLVED__"

            user_ref.set(
                {
                    "known_slots": known_slots,
                    "ui_state": "none",
                    "conversation_mode": "consulting",
                    "recent_quick_reply_count": 0,
                },
                merge=True,
            )
            msg = unresolved_slot_response(slot)
            reply_and_remember(reply_token, user_ref, user_data, msg)
            return

        known_slots = apply_quick_reply_selection(user_data, data)
        current_topic = user_data.get("current_topic") or user_data.get("last_topic") or "relationship"
        active_text = user_data.get("active_consultation_text") or user_data.get("last_consultation_text") or ""
        plan_tier = user_data.get("plan_tier", "free")
        needed = required_slots_for_topic(current_topic, plan_tier)
        filled = [k for k in needed if k in known_slots]
        missing = [s for s in needed if s not in known_slots]

        user_ref.set(
            {
                "known_slots": known_slots,
                "ui_state": "none",
                "conversation_mode": "consulting",
                "recent_quick_reply_count": 0,
                "last_user_ignored_quick_reply": False,
            },
            merge=True,
        )

        if len(filled) >= min(3, len(needed)):
            bridge = choose_variant("reading_bridge", user_data)
            save_variant_state(user_ref, "reading_bridge", user_data)
            reply_body, oracle_result, context_feats = build_reading_reply(
                user_data={**user_data, "current_topic": current_topic},
                active_text=active_text or current_topic,
                known_slots=known_slots,
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
                        "recent_quick_reply_count": 1,
                    },
                    merge=True,
                )
                reply_and_remember(reply_token, user_ref, user_data, f"{bridge}\n\n{reply_body}", quick_reply=birth_qr)
                return

            user_ref.set(
                {
                    "conversation_mode": "post_oracle",
                    "last_oracle_message": oracle_result["message"],
                    "last_oracle_summary": oracle_result["summary"],
                    "last_context": context_feats,
                    "last_topic": oracle_result["topic"],
                },
                merge=True,
            )
            reply_and_remember(reply_token, user_ref, user_data, f"{bridge}\n\n{reply_body}")
            return

        if missing and should_offer_quick_reply(user_data, missing, turn_meta={"intent": "continue_consult", "turn_zone": "consultation"}):
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
                    "recent_quick_reply_count": 1,
                },
                merge=True,
            )
            reply_and_remember(reply_token, user_ref, user_data, prompt, quick_reply=qr)
            return

        msg = "そのまま続けて話してください。必要な輪郭は少しずつ見えてきています。"
        reply_and_remember(reply_token, user_ref, user_data, msg)
        return

    msg = "そのまま続けてください。必要なところだけ、こちらで拾っていきます。"
    reply_and_remember(reply_token, user_ref, user_data, msg)


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
            msg_text = gemini_text(prompt, "新しい朝が来ました。そのままのあなたで。――識より")
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
        user_data = implicit_suspend_check(user_ref, user_data)
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
