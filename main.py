import os
import json
import logging
import re
import unicodedata
from datetime import datetime, timezone

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage
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

        logger.info("Firebase project_id: %s", key_dict.get("project_id"))
        logger.info("Firebase client_email: %s", key_dict.get("client_email"))
        logger.info("Firebase private_key_id: %s", key_dict.get("private_key_id"))
        logger.info(
            "Firebase private_key format OK: %s",
            str(key_dict.get("private_key", "")).startswith("-----BEGIN PRIVATE KEY-----")
        )

        cred = credentials.Certificate(key_dict)
        firebase_admin.initialize_app(cred)
        logger.info("Firebase initialization successful using env JSON.")
    except Exception as e:
        logger.exception("Firebase initialization failed")
        raise RuntimeError(f"Firebase initialization failed: {e}")

db = firestore.client()


# -------------------------
# 生年月日パース
# -------------------------
def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text.strip())


def parse_japanese_era_date(text: str):
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


def parse_birth_date(text: str):
    text = normalize_text(text)

    era_result = parse_japanese_era_date(text)
    if era_result:
        return era_result

    m = re.match(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$", text)
    if m:
        year, month, day = map(int, m.groups())
        try:
            dt = datetime(year, month, day)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None

    m = re.match(r"^(\d{4})年(\d{1,2})月(\d{1,2})日$", text)
    if m:
        year, month, day = map(int, m.groups())
        try:
            dt = datetime(year, month, day)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None

    m = re.match(r"^(\d{4})(\d{2})(\d{2})$", text)
    if m:
        year, month, day = map(int, m.groups())
        try:
            dt = datetime(year, month, day)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None

    m = re.match(r"^(\d{4})(\d{1,2})(\d{1,2})$", text)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        day = int(m.group(3))
        try:
            dt = datetime(year, month, day)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None

    return None


def looks_like_birth_date(text: str) -> bool:
    return parse_birth_date(text) is not None


# -------------------------
# 補助関数
# -------------------------
def build_user_profile(user_data: dict) -> dict:
    user_profile = {
        "birth_year": 1990,
        "birth_month": 6,
        "birth_day": 15,
        "resilience": float(user_data.get("resilience", 0.55)),
        "sensitivity": float(user_data.get("sensitivity", 0.70)),
        "patience": float(user_data.get("patience", 0.45))
    }

    birth_date = user_data.get("birth_date")
    if birth_date:
        try:
            y, m, d = birth_date.split("-")
            user_profile["birth_year"] = int(y)
            user_profile["birth_month"] = int(m)
            user_profile["birth_day"] = int(d)
        except Exception:
            pass

    return user_profile


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


def required_slots_for_topic(topic: str, is_paid: bool) -> list:
    common = ["time_continuity", "emotion"]
    topic_map = {
        "love": ["relationship_distance", "desired_action"],
        "work": ["main_stressor", "desired_action"],
        "relationship": ["person_type", "desired_action"],
    }
    slots = common + topic_map.get(topic, [])
    if is_paid:
        slots.append("detail_depth")
    return slots


def merge_known_slots(existing: dict, new_data: dict) -> dict:
    merged = dict(existing or {})
    for k, v in (new_data or {}).items():
        if v is not None and v != "":
            merged[k] = v
    return merged


def slots_to_context(base_context: dict, known_slots: dict) -> dict:
    updated = dict(base_context or {})
    known_slots = known_slots or {}

    emotion = str(known_slots.get("emotion", ""))
    continuity = str(known_slots.get("time_continuity", ""))
    desired_action = str(known_slots.get("desired_action", ""))
    relationship_distance = str(known_slots.get("relationship_distance", ""))
    main_stressor = str(known_slots.get("main_stressor", ""))
    detail_depth = str(known_slots.get("detail_depth", ""))

    if any(x in emotion for x in ["焦", "不安", "ソワ", "落ち着かない"]):
        updated["urgency"] = min(float(updated.get("urgency", 0.5)) + 0.15, 1.0)
    if any(x in emotion for x in ["悲", "寂", "孤独", "つらい"]):
        updated["loneliness"] = min(float(updated.get("loneliness", 0.5)) + 0.15, 1.0)
    if any(x in emotion for x in ["怒", "イライラ", "腹立"]):
        updated["stress"] = min(float(updated.get("stress", 0.5)) + 0.15, 1.0)
    if any(x in emotion for x in ["何も感じない", "空虚", "虚しい", "無"]):
        updated["stress"] = min(float(updated.get("stress", 0.5)) + 0.10, 1.0)
        updated["loneliness"] = min(float(updated.get("loneliness", 0.5)) + 0.10, 1.0)

    if any(x in continuity for x in ["前から", "ずっと", "長い", "しばらく", "続いて"]):
        updated["stress"] = min(float(updated.get("stress", 0.5)) + 0.05, 1.0)

    if any(x in desired_action for x in ["待", "様子見", "少し置く"]):
        updated["urgency"] = max(float(updated.get("urgency", 0.5)) - 0.10, 0.0)
    if any(x in desired_action for x in ["動きたい", "連絡", "伝えたい", "進めたい"]):
        updated["urgency"] = min(float(updated.get("urgency", 0.5)) + 0.10, 1.0)

    if any(x in relationship_distance for x in ["かなり離", "遠い", "既読無視", "返事がない"]):
        updated["loneliness"] = min(float(updated.get("loneliness", 0.5)) + 0.10, 1.0)

    if any(x in main_stressor for x in ["仕事量", "忙し", "残業"]):
        updated["stress"] = min(float(updated.get("stress", 0.5)) + 0.12, 1.0)
    if any(x in main_stressor for x in ["人間関係", "上司", "同僚"]):
        updated["stress"] = min(float(updated.get("stress", 0.5)) + 0.10, 1.0)
        updated["loneliness"] = min(float(updated.get("loneliness", 0.5)) + 0.05, 1.0)

    if any(x in detail_depth for x in ["眠れてない", "休めてない", "限界", "しんどい"]):
        updated["sleep_deficit"] = min(float(updated.get("sleep_deficit", 0.5)) + 0.15, 1.0)
        updated["stress"] = min(float(updated.get("stress", 0.5)) + 0.10, 1.0)

    return updated


def extract_json_from_text(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    codeblock = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if codeblock:
        return json.loads(codeblock.group(1))

    brace = re.search(r"(\{.*\})", text, re.DOTALL)
    if brace:
        return json.loads(brace.group(1))

    raise ValueError("JSONを抽出できませんでした")


def call_gemini_json(prompt: str) -> dict:
    response = model.generate_content(prompt)
    raw_text = getattr(response, "text", "").strip()
    return extract_json_from_text(raw_text)


# -------------------------
# 会話分析（神託前）
# -------------------------
def analyze_conversation_with_gemini(
    user_text: str,
    current_topic: str | None,
    known_slots: dict | None,
    last_question: str | None,
    is_paid: bool
) -> dict:
    known_slots = known_slots or {}
    topic_hint = current_topic or "unknown"

    prompt = f"""
あなたは会話整理専用の判定器です。
必ずJSONのみを返してください。説明文は禁止です。

目的:
- ユーザーの今回の発言から、相談テーマ、取得済み情報、足りない情報、次に自然に聞くことを整理する
- 前の質問に直接答えていなくても、意味のある補足は拾う
- 相談が明らかに別テーマへ変わったら switched_topic=true にする
- 生年月日らしき入力はここでは扱わない（別処理済み）

出力ルール:
- JSONのみ
- topic は love / work / relationship
- extracted_slots は今回の発言から拾えた情報だけ
- next_question は自然な日本語
- should_generate_reading は、もう神託を返してよいなら true

現在のテーマ候補:
{topic_hint}

現在の既知スロット:
{json.dumps(known_slots, ensure_ascii=False)}

前回ユーザーに聞いた質問:
{last_question or ""}

今回のユーザー発言:
{user_text}

参考スロット:
time_continuity, emotion, desired_action, relationship_distance, main_stressor, person_type, detail_depth

テーマ別の最低必要情報:
- love: time_continuity, emotion, relationship_distance, desired_action
- work: time_continuity, emotion, main_stressor, desired_action
- relationship: time_continuity, emotion, person_type, desired_action

JSON形式:
{{
  "topic": "love",
  "answered_previous_question": true,
  "switched_topic": false,
  "extracted_slots": {{
    "time_continuity": "前からずっと続いている",
    "emotion": "不安が強い"
  }},
  "missing_slots": ["relationship_distance", "desired_action"],
  "should_generate_reading": false,
  "next_question": "相手との距離は今どうですか？ 近い感じですか、それとも少し離れていますか？",
  "bridge_text": "その揺れは一時的なものではなさそうですね。"
}}
""".strip()

    data = call_gemini_json(prompt)

    if data.get("topic") not in ["love", "work", "relationship"]:
        data["topic"] = current_topic or "relationship"

    if not isinstance(data.get("extracted_slots"), dict):
        data["extracted_slots"] = {}

    if not isinstance(data.get("missing_slots"), list):
        data["missing_slots"] = []

    if not isinstance(data.get("should_generate_reading"), bool):
        data["should_generate_reading"] = False

    if not isinstance(data.get("answered_previous_question"), bool):
        data["answered_previous_question"] = False

    if not isinstance(data.get("switched_topic"), bool):
        data["switched_topic"] = False

    if not isinstance(data.get("next_question"), str):
        data["next_question"] = ""

    if not isinstance(data.get("bridge_text"), str):
        data["bridge_text"] = ""

    return data


def fallback_conversation_analysis(user_text: str, current_topic: str | None, known_slots: dict | None, is_paid: bool) -> dict:
    topic = current_topic or oracle_engine.topic_classifier.classify(user_text)
    known_slots = known_slots or {}
    extracted = {}

    txt = normalize_text(user_text)

    if any(x in txt for x in ["前から", "ずっと", "しばらく", "長く続いて"]):
        extracted["time_continuity"] = "前から続いている"
    elif any(x in txt for x in ["急に", "突然", "最近いきなり"]):
        extracted["time_continuity"] = "急に強くなった"

    if any(x in txt for x in ["不安", "焦り", "こわい", "怖い"]):
        extracted["emotion"] = "不安や焦りが強い"
    elif any(x in txt for x in ["悲しい", "寂しい", "孤独", "つらい"]):
        extracted["emotion"] = "悲しさや寂しさが強い"
    elif any(x in txt for x in ["イライラ", "怒", "腹立"]):
        extracted["emotion"] = "怒りやイライラが強い"

    if any(x in txt for x in ["待ちたい", "様子見", "少し置きたい"]):
        extracted["desired_action"] = "待ちたい"
    elif any(x in txt for x in ["連絡したい", "動きたい", "伝えたい", "進めたい"]):
        extracted["desired_action"] = "動きたい"

    if topic == "love":
        if any(x in txt for x in ["既読無視", "返事がない", "距離がある", "離れてる"]):
            extracted["relationship_distance"] = "少し離れている"
    elif topic == "work":
        if any(x in txt for x in ["上司", "同僚", "人間関係"]):
            extracted["main_stressor"] = "人間関係"
        elif any(x in txt for x in ["仕事量", "忙しい", "残業"]):
            extracted["main_stressor"] = "仕事量"
    else:
        if any(x in txt for x in ["家族", "親", "母", "父"]):
            extracted["person_type"] = "家族"
        elif any(x in txt for x in ["友達", "友人"]):
            extracted["person_type"] = "友人"
        elif any(x in txt for x in ["職場", "上司", "同僚"]):
            extracted["person_type"] = "職場"
        elif any(x in txt for x in ["彼", "彼女", "恋人", "パートナー"]):
            extracted["person_type"] = "恋人"

    merged = merge_known_slots(known_slots, extracted)
    required = required_slots_for_topic(topic, is_paid)
    missing = [s for s in required if not merged.get(s)]

    question_map = {
        "time_continuity": "その悩みって、急に強くなった感じですか？ それとも前からずっと続いていましたか？",
        "emotion": "今の気持ちにいちばん近いのはどれですか？ 焦り / 悲しさ / イライラ / 何も感じない感じ",
        "relationship_distance": "相手との距離は今どうですか？ 近い感じですか、それとも少し離れていますか？",
        "desired_action": "今は動きたいですか？ それとも少し様子を見たい気持ちの方が近いですか？",
        "main_stressor": "いちばんしんどいのはどれに近いですか？ 仕事量 / 人間関係 / 将来の不安",
        "person_type": "その相手は誰に近いですか？ 家族 / 友人 / 職場 / 恋人",
        "detail_depth": "最近のあなたは、ちゃんと休めていますか？ それともかなり消耗していますか？",
    }

    return {
        "topic": topic,
        "answered_previous_question": True,
        "switched_topic": False,
        "extracted_slots": extracted,
        "missing_slots": missing,
        "should_generate_reading": len(missing) == 0,
        "next_question": question_map.get(missing[0], "") if missing else "",
        "bridge_text": "少しずつ輪郭が見えてきました。"
    }


def get_conversation_analysis(user_text: str, current_topic: str | None, known_slots: dict | None, last_question: str | None, is_paid: bool) -> dict:
    try:
        return analyze_conversation_with_gemini(
            user_text=user_text,
            current_topic=current_topic,
            known_slots=known_slots,
            last_question=last_question,
            is_paid=is_paid
        )
    except Exception:
        logger.exception("Gemini conversation analysis failed, using fallback")
        return fallback_conversation_analysis(user_text, current_topic, known_slots, is_paid)


# -------------------------
# 神託後の意図判定
# -------------------------
def analyze_post_oracle_with_gemini(user_text: str, last_oracle_message: str, oracle_summary: dict) -> dict:
    prompt = f"""
あなたは神託後の会話意図を判定する分類器です。
必ずJSONのみを返してください。説明は禁止です。

直前の神託:
{last_oracle_message}

内部要約:
{json.dumps(oracle_summary, ensure_ascii=False)}

ユーザー発言:
{user_text}

分類したい意図:
- clarify_oracle: 神託をわかりやすく説明してほしい
- ask_action_guidance: 結局どう動けばいいか聞いている
- deepen_current_topic: 同じ相談をもっと深く見たい
- emotional_reaction: 感情の吐露や反応
- disagree_oracle: 納得できない、違う気がする
- switch_new_consult: 新しい相談へ切り替えたい
- other

出力形式:
{{
  "intent": "clarify_oracle",
  "refers_to_last_oracle": true,
  "needs_new_prediction": false,
  "reply_bridge": "わかりにくかったですね。もう少しかみ砕いて返します。"
}}
""".strip()

    data = call_gemini_json(prompt)

    allowed = {
        "clarify_oracle",
        "ask_action_guidance",
        "deepen_current_topic",
        "emotional_reaction",
        "disagree_oracle",
        "switch_new_consult",
        "other"
    }

    if data.get("intent") not in allowed:
        data["intent"] = "other"

    if not isinstance(data.get("refers_to_last_oracle"), bool):
        data["refers_to_last_oracle"] = True

    if not isinstance(data.get("needs_new_prediction"), bool):
        data["needs_new_prediction"] = False

    if not isinstance(data.get("reply_bridge"), str):
        data["reply_bridge"] = ""

    return data


def fallback_post_oracle_intent(user_text: str) -> dict:
    txt = normalize_text(user_text)

    if any(x in txt for x in ["どういうこと", "わかりやすく", "簡単に", "つまり", "意味"]):
        intent = "clarify_oracle"
    elif any(x in txt for x in ["どうしたら", "何をすれば", "結局", "行動", "連絡していい", "待てばいい"]):
        intent = "ask_action_guidance"
    elif any(x in txt for x in ["もっと詳しく", "深く", "相手の気持ち", "今週", "先も見て"]):
        intent = "deepen_current_topic"
    elif any(x in txt for x in ["違う", "違う気がする", "当たってない", "そんなことない"]):
        intent = "disagree_oracle"
    elif any(x in txt for x in ["ちなみに", "別件", "それとは別に", "仕事も", "家族も", "他にも"]):
        intent = "switch_new_consult"
    elif any(x in txt for x in ["つらい", "しんどい", "不安", "安心した", "少し楽"]):
        intent = "emotional_reaction"
    else:
        intent = "other"

    return {
        "intent": intent,
        "refers_to_last_oracle": True,
        "needs_new_prediction": False,
        "reply_bridge": ""
    }


def get_post_oracle_intent(user_text: str, last_oracle_message: str, oracle_summary: dict) -> dict:
    try:
        return analyze_post_oracle_with_gemini(user_text, last_oracle_message, oracle_summary)
    except Exception:
        logger.exception("Gemini post-oracle intent failed, using fallback")
        return fallback_post_oracle_intent(user_text)


# -------------------------
# 神託後の返答生成
# -------------------------
def explain_oracle_simple(user_text: str, last_oracle_message: str, oracle_summary: dict) -> str:
    prompt = f"""
あなたは神秘的な存在『識（SHIKI）』です。
ただし今回は新しい神託を作らず、直前に返した神託をわかりやすい言葉で説明してください。
説教しないでください。
長すぎず、2〜4文程度で、日本語で自然に返してください。
神託の文面をそのまま繰り返さず、意味をかみ砕いてください。

直前の神託:
{last_oracle_message}

内部要約:
{json.dumps(oracle_summary, ensure_ascii=False)}

ユーザーの聞き方:
{user_text}
""".strip()

    try:
        response = model.generate_content(prompt)
        text = getattr(response, "text", "").strip()
        if text:
            return text
    except Exception:
        logger.exception("explain_oracle_simple failed")

    return (
        f"簡単に言うと、{oracle_summary.get('core_meaning', '今は流れを急いで掴みにいくより、整えながら見た方がいい時です。')}"
        f"\n{oracle_summary.get('risk_hint', '')}"
    )


def explain_oracle_action(user_text: str, last_oracle_message: str, oracle_summary: dict) -> str:
    prompt = f"""
あなたは神秘的な存在『識（SHIKI）』です。
ただし今回は新しい神託を作らず、直前の神託を行動レベルに落として返してください。
ユーザーが「結局どうしたらいいのか」を知りたがっています。
2〜4文程度で、日本語で自然に返してください。
直前の神託をそのまま繰り返さないでください。

直前の神託:
{last_oracle_message}

内部要約:
{json.dumps(oracle_summary, ensure_ascii=False)}

ユーザーの聞き方:
{user_text}
""".strip()

    try:
        response = model.generate_content(prompt)
        text = getattr(response, "text", "").strip()
        if text:
            return text
    except Exception:
        logger.exception("explain_oracle_action failed")

    return oracle_summary.get("action_hint", "今は答えを急ぐより、ひとつだけ整えてから次を見る方がよさそうです。")


def respond_to_emotion(user_text: str, last_oracle_message: str, oracle_summary: dict) -> str:
    prompt = f"""
あなたは神秘的な存在『識（SHIKI）』です。
今回は新しい神託ではなく、直前の神託に対するユーザーの感情反応に寄り添って返してください。
やさしく、短く、自然な日本語で2〜4文。
説教しないこと。神託をそのまま繰り返さないこと。

直前の神託:
{last_oracle_message}

内部要約:
{json.dumps(oracle_summary, ensure_ascii=False)}

ユーザーの反応:
{user_text}
""".strip()

    try:
        response = model.generate_content(prompt)
        text = getattr(response, "text", "").strip()
        if text:
            return text
    except Exception:
        logger.exception("respond_to_emotion failed")

    return "その反応も自然です。いまは答えを急ぐより、自分の感覚がどう揺れているかを見失わないことの方が大事です。"


def respond_to_disagreement(user_text: str, last_oracle_message: str, oracle_summary: dict) -> str:
    prompt = f"""
あなたは神秘的な存在『識（SHIKI）』です。
ユーザーは直前の神託に違和感を持っています。
言い返さず、守りに入らず、やわらかく受け止めてください。
そのうえで、どの部分がズレているのかを一言聞き返す形で返してください。
2〜4文、日本語で自然に。

直前の神託:
{last_oracle_message}

内部要約:
{json.dumps(oracle_summary, ensure_ascii=False)}

ユーザーの反応:
{user_text}
""".strip()

    try:
        response = model.generate_content(prompt)
        text = getattr(response, "text", "").strip()
        if text:
            return text
    except Exception:
        logger.exception("respond_to_disagreement failed")

    return "ズレを感じたなら、それは大事な反応です。どの部分がいちばん違うと感じたのか、そこだけ教えてください。"


def pre_oracle_bridge_text(known_slots: dict) -> str:
    if known_slots.get("emotion") and known_slots.get("desired_action"):
        return "だいぶ輪郭が見えてきました。では、今のあなたに近い流れを言葉にします。"
    return "少しずつ流れの形が見えてきました。では、今視えているものを返します。"


# -------------------------
# 相談分析
# -------------------------
def build_reading_reply(user_data: dict, active_text: str, topic: str, known_slots: dict, is_paid: bool):
    base_context = user_data.get("last_context") or build_base_context(user_data)
    context_feats = slots_to_context(base_context, known_slots)
    user_profile = build_user_profile(user_data)
    memory = build_memory(user_data)

    horizon = "week" if is_paid else "today"

    oracle_result = oracle_engine.predict(
        user_profile=user_profile,
        context_feats=context_feats,
        user_text=active_text,
        date_str=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        horizon=horizon,
        memory=memory,
        is_paid=is_paid
    )

    reply_text = oracle_result["message"]

    if not user_data.get("birth_date"):
        reply_text += (
            "\n\nもっと深く視るには、生まれた日の流れも重ねた方が精度が上がります。"
            "\n生年月日は、1990-05-12、1990年5月12日、昭和60年5月12日 みたいな形で送ってもらえれば大丈夫です。"
        )

    return reply_text, oracle_result, context_feats


# -------------------------
# 朝通知用
# -------------------------
def generate_mystical_message(user_text: str) -> str:
    prompt = (
        "あなたは神秘的な存在『識（SHIKI）』。孤独を肯定し、静かに寄り添います。"
        f"\nユーザーの昨日の言葉：『{user_text}』"
        "\nこの言葉を元に、今日を歩むための占い的な一言を80文字以内で作成してください。"
        "\n言い回しが毎回同じにならないようにしてください。"
        "\n語尾に必ず『――識より』を添えて。"
    )

    try:
        response = model.generate_content(prompt)
        text = getattr(response, "text", None)
        if text and text.strip():
            return text.strip()
        return "新しい朝が来ました。そのままのあなたで。――識より"
    except Exception as e:
        logger.error("Gemini morning error: %s", e)
        return "新しい朝が来ました。そのままのあなたで。――識より"


# -------------------------
# API
# -------------------------
@app.get("/")
def root():
    return {
        "status": "online",
        "message": "SHIKI System is running."
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "firebase_initialized": bool(firebase_admin._apps),
        "gemini_model": GEMINI_MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat()
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

            msg_text = generate_mystical_message(last_msg)
            line_bot_api.push_message(u_id, TextSendMessage(text=msg_text))
            count += 1

        logger.info("Morning push completed. sent_count=%s", count)
        return {"status": "completed", "sent_count": count}

    except Exception as e:
        logger.exception("Push error")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


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


@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    try:
        u_id = event.source.user_id
        u_text = event.message.text.strip()

        user_ref = db.collection("users").document(u_id)
        user_doc = user_ref.get()
        user_data = user_doc.to_dict() or {}

        user_ref.set(
            {
                "last_msg": u_text,
                "last_active": datetime.now(timezone.utc)
            },
            merge=True
        )

        is_paid = bool(user_data.get("is_paid", False))
        conversation_mode = user_data.get("conversation_mode", "idle")

        # --------------------------------
        # 1. 生年月日入力
        # --------------------------------
        if looks_like_birth_date(u_text):
            parsed_birth = parse_birth_date(u_text)
            if parsed_birth:
                updated_user_data = {**user_data, "birth_date": parsed_birth}
                user_ref.set({"birth_date": parsed_birth}, merge=True)

                active_text = user_data.get("active_consultation_text") or user_data.get("last_consultation_text")
                current_topic = user_data.get("current_topic") or user_data.get("last_topic")
                known_slots = user_data.get("known_slots", {}) or {}

                if active_text and current_topic:
                    reply_text, oracle_result, context_feats = build_reading_reply(
                        user_data=updated_user_data,
                        active_text=active_text,
                        topic=current_topic,
                        known_slots=known_slots,
                        is_paid=is_paid
                    )

                    user_ref.set(
                        {
                            "last_topic": oracle_result["topic"],
                            "last_oracle_message": oracle_result["message"],
                            "last_oracle_summary": oracle_result["summary"],
                            "oracle_engine_version": oracle_result["engine_version"],
                            "last_context": context_feats,
                            "conversation_mode": "post_oracle"
                        },
                        merge=True
                    )

                    line_bot_api.reply_message(
                        event.reply_token,
                        TextSendMessage(
                            text=(
                                f"生まれた日の気配を受け取りました。{parsed_birth} として記録しておきます。\n\n"
                                f"さっきの流れを、その巡りも重ねてもう一度視ました。\n"
                                f"{reply_text}\n\n"
                                f"もし前に預けた内容が違っていた場合も、今回の内容で上書きされています。"
                            )
                        )
                    )
                    return

                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(
                        text=(
                            f"生まれた日の気配を受け取りました。{parsed_birth} として記録しておきます。\n"
                            f"もし前に預けた内容が違っていた場合も、今回の内容で上書きされています。"
                        )
                    )
                )
                return

        # --------------------------------
        # 2. 神託後モード
        # --------------------------------
        last_oracle_message = user_data.get("last_oracle_message")
        last_oracle_summary = user_data.get("last_oracle_summary") or {}

        if conversation_mode == "post_oracle" and last_oracle_message:
            intent_data = get_post_oracle_intent(
                user_text=u_text,
                last_oracle_message=last_oracle_message,
                oracle_summary=last_oracle_summary
            )
            intent = intent_data.get("intent", "other")

            if intent == "clarify_oracle":
                reply_text = explain_oracle_simple(u_text, last_oracle_message, last_oracle_summary)
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
                return

            if intent == "ask_action_guidance":
                reply_text = explain_oracle_action(u_text, last_oracle_message, last_oracle_summary)
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
                return

            if intent == "emotional_reaction":
                reply_text = respond_to_emotion(u_text, last_oracle_message, last_oracle_summary)
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
                return

            if intent == "disagree_oracle":
                reply_text = respond_to_disagreement(u_text, last_oracle_message, last_oracle_summary)
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
                return

            if intent == "deepen_current_topic":
                current_topic = user_data.get("current_topic") or user_data.get("last_topic") or "relationship"
                known_slots = user_data.get("known_slots", {}) or {}
                missing_slots = required_slots_for_topic(current_topic, True)  # 深掘りは有料前提に近い聞き方
                missing_slots = [s for s in missing_slots if not known_slots.get(s)]

                question_map = {
                    "time_continuity": "その悩みって、急に強くなった感じですか？ それとも前からずっと続いていましたか？",
                    "emotion": "今の気持ちにいちばん近いのはどれですか？ 焦り / 悲しさ / イライラ / 何も感じない感じ",
                    "relationship_distance": "相手との距離は今どうですか？ 近い感じですか、それとも少し離れていますか？",
                    "desired_action": "今は動きたいですか？ それとも少し様子を見たい気持ちの方が近いですか？",
                    "main_stressor": "いちばんしんどいのはどれに近いですか？ 仕事量 / 人間関係 / 将来の不安",
                    "person_type": "その相手は誰に近いですか？ 家族 / 友人 / 職場 / 恋人",
                    "detail_depth": "最近のあなたは、ちゃんと休めていますか？ それともかなり消耗していますか？",
                }

                next_question = question_map.get(missing_slots[0], "もう少しだけ、今の流れを深く見るために教えてください。") if missing_slots else "もう少し深く見るために、今いちばん引っかかっている部分をそのまま話してください。"

                user_ref.set(
                    {
                        "conversation_mode": "consulting",
                        "last_question": next_question,
                        "active_consultation_text": user_data.get("active_consultation_text") or user_data.get("last_consultation_text"),
                        "current_topic": current_topic
                    },
                    merge=True
                )

                bridge = intent_data.get("reply_bridge") or "では、さっきの流れをもう少し深く見ます。"
                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text=f"{bridge}\n{next_question}")
                )
                return

            if intent == "switch_new_consult":
                user_ref.set(
                    {
                        "conversation_mode": "consulting",
                        "current_topic": firestore.DELETE_FIELD,
                        "active_consultation_text": firestore.DELETE_FIELD,
                        "known_slots": firestore.DELETE_FIELD,
                        "last_question": firestore.DELETE_FIELD
                    },
                    merge=True
                )
                # 下の新規相談処理へ流す

            else:
                # other は、まず前神託に対する軽い応答を返して様子を見る
                reply_text = explain_oracle_simple(u_text, last_oracle_message, last_oracle_summary)
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
                return

        # --------------------------------
        # 3. 相談進行中または新規相談
        # --------------------------------
        current_topic = user_data.get("current_topic")
        active_consultation_text = user_data.get("active_consultation_text")
        known_slots = user_data.get("known_slots", {}) or {}
        last_question = user_data.get("last_question")

        if not active_consultation_text:
            current_topic = oracle_engine.topic_classifier.classify(u_text)
            active_consultation_text = u_text
            known_slots = {}
            last_question = None

            user_ref.set(
                {
                    "conversation_mode": "consulting",
                    "current_topic": current_topic,
                    "active_consultation_text": active_consultation_text,
                    "known_slots": known_slots,
                    "last_consultation_text": u_text
                },
                merge=True
            )

            intake_opening = "その迷い、たしかに受け取りました。まだ答えを急がず、少しだけ流れの輪郭を確かめさせてください。"
            analysis = get_conversation_analysis(
                user_text=u_text,
                current_topic=current_topic,
                known_slots=known_slots,
                last_question=None,
                is_paid=is_paid
            )

            known_slots = merge_known_slots(known_slots, analysis.get("extracted_slots", {}))
            required = required_slots_for_topic(current_topic, is_paid)
            missing_slots = [s for s in required if not known_slots.get(s)]
            should_generate = analysis.get("should_generate_reading", False) or len(missing_slots) == 0

            if not should_generate:
                next_question = analysis.get("next_question", "").strip()
                if not next_question:
                    next_question = "まずひとつだけ聞かせてください。今の気持ちにいちばん近いのは、不安・悲しさ・イライラ・何も感じない感じのどれですか？"

                user_ref.set(
                    {
                        "conversation_mode": "consulting",
                        "current_topic": current_topic,
                        "active_consultation_text": active_consultation_text,
                        "known_slots": known_slots,
                        "missing_slots": missing_slots,
                        "last_question": next_question
                    },
                    merge=True
                )

                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text=f"{intake_opening}\n{next_question}")
                )
                return

            bridge = "だいぶ輪郭が見えてきました。では、今のあなたに近い流れを言葉にします。"
            reply_text, oracle_result, context_feats = build_reading_reply(
                user_data={**user_data, "known_slots": known_slots},
                active_text=active_consultation_text,
                topic=current_topic,
                known_slots=known_slots,
                is_paid=is_paid
            )

            final_text = f"{bridge}\n\n{reply_text}"

            user_ref.set(
                {
                    "conversation_mode": "post_oracle",
                    "current_topic": current_topic,
                    "active_consultation_text": active_consultation_text,
                    "known_slots": known_slots,
                    "missing_slots": [],
                    "last_question": firestore.DELETE_FIELD,
                    "last_topic": oracle_result["topic"],
                    "last_oracle_message": oracle_result["message"],
                    "last_oracle_summary": oracle_result["summary"],
                    "oracle_engine_version": oracle_result["engine_version"],
                    "last_consultation_text": active_consultation_text,
                    "last_context": context_feats
                },
                merge=True
            )

            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=final_text))
            return

        # 既存相談の続き
        analysis = get_conversation_analysis(
            user_text=u_text,
            current_topic=current_topic,
            known_slots=known_slots,
            last_question=last_question,
            is_paid=is_paid
        )

        if analysis.get("switched_topic"):
            current_topic = analysis["topic"]
            active_consultation_text = u_text
            known_slots = {}
            last_question = None

        extracted_slots = analysis.get("extracted_slots", {})
        known_slots = merge_known_slots(known_slots, extracted_slots)

        required = required_slots_for_topic(current_topic, is_paid)
        missing_slots = [s for s in required if not known_slots.get(s)]
        should_generate = analysis.get("should_generate_reading", False) or len(missing_slots) == 0

        if not should_generate:
            next_question = analysis.get("next_question", "").strip()
            bridge_text = analysis.get("bridge_text", "").strip()

            if not next_question:
                next_question = "もう少しだけ聞かせてください。"

            user_ref.set(
                {
                    "conversation_mode": "consulting",
                    "current_topic": current_topic,
                    "active_consultation_text": active_consultation_text,
                    "known_slots": known_slots,
                    "missing_slots": missing_slots,
                    "last_question": next_question
                },
                merge=True
            )

            reply_text = f"{bridge_text}\n{next_question}" if bridge_text else next_question
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
            return

        # 神託生成
        bridge = pre_oracle_bridge_text(known_slots)
        reply_text, oracle_result, context_feats = build_reading_reply(
            user_data={**user_data, "known_slots": known_slots},
            active_text=active_consultation_text,
            topic=current_topic,
            known_slots=known_slots,
            is_paid=is_paid
        )

        final_text = f"{bridge}\n\n{reply_text}"

        user_ref.set(
            {
                "conversation_mode": "post_oracle",
                "current_topic": current_topic,
                "active_consultation_text": active_consultation_text,
                "known_slots": known_slots,
                "missing_slots": [],
                "last_question": firestore.DELETE_FIELD,
                "last_topic": oracle_result["topic"],
                "last_oracle_message": oracle_result["message"],
                "last_oracle_summary": oracle_result["summary"],
                "oracle_engine_version": oracle_result["engine_version"],
                "last_consultation_text": active_consultation_text,
                "last_context": context_feats
            },
            merge=True
        )

        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=final_text))

    except Exception as e:
        logger.exception("handle_text error")
        try:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="識の観測にわずかな乱れが生じました。少し時間を置いてもう一度声をかけてください。")
            )
        except Exception:
            logger.exception("Reply fallback failed")
