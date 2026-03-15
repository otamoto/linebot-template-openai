import os
import json
import logging
import threading
import random
import re
import unicodedata
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from urllib.parse import parse_qsl
from uuid import uuid4

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from linebot import LineBotApi, WebhookHandler
from linebot.models import (
    MessageEvent,
    TextMessage,
    TextSendMessage,
    QuickReply,
    QuickReplyButton,
    PostbackAction,
    PostbackEvent,
    DatetimePickerAction,
    TemplateSendMessage,
    ButtonsTemplate,
)
from linebot.exceptions import InvalidSignatureError, LineBotApiError

from openai import OpenAI

import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin.firestore import DELETE_FIELD

from oracle_engine import OracleEngine


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s : %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="SHIKI LINE Bot")


PHASE_WAIT_INITIAL_CONSULT = "waiting_initial_consult"
PHASE_WAIT_NAME = "waiting_name"
PHASE_WAIT_BIRTH_DATE = "waiting_birth_date"
PHASE_WAIT_BIRTH_TIME = "waiting_birth_time"
PHASE_WAIT_BIRTH_PREFECTURE = "waiting_birth_prefecture"
PHASE_WAIT_PROFILE_CONFIRM = "waiting_profile_confirm"
PHASE_WAIT_RESTART_CONFIRM = "waiting_restart_confirm"
PHASE_WAIT_CONSULT_DETAIL = "waiting_consult_detail"
PHASE_WAIT_MOTIF = "waiting_motif"
PHASE_WAIT_COLDREAD_RESPONSE = "waiting_coldread_response"
PHASE_WAIT_FOLLOWUP_MENU = "waiting_followup_menu"
PHASE_DIALOGUE = "dialogue"
PHASE_WAIT_PAYMENT = "waiting_payment"

PLAN_FREE = "free"
PLAN_PAID = "paid"
PLAN_PREMIUM = "premium"

DEFAULT_UNKNOWN_HOUR = 12
DEFAULT_UNKNOWN_MINUTE = 0
DEFAULT_UNKNOWN_SECOND = 0
DEFAULT_BIRTH_LONGITUDE = float(os.getenv("DEFAULT_BIRTH_LONGITUDE", "135.0"))

MAX_CHAT_HISTORY_CHARS = 5000
PROFILE_DEFAULT_NAME = "あなた"

DEFAULT_FREE_SESSIONS = int(os.getenv("DEFAULT_FREE_SESSIONS", "1"))
PAYMENT_URL = os.getenv("PAYMENT_URL", "https://example.com/payment")
PREMIUM_PAYMENT_URL = os.getenv("PREMIUM_PAYMENT_URL", PAYMENT_URL)
SERVICE_NAME = os.getenv("SERVICE_NAME", "SHIKI")

ALL_MOTIFS = [
    "銀の鍵", "砂時計", "古びた鏡", "聖なる滴", "琥珀の蝶", "折れた剣", "青い月", "羅針盤", "封じられた手紙", "金の天秤",
    "揺れる灯火", "水晶の髑髏", "沈黙の鐘", "茨の冠", "無垢な羽根", "双子の蛇", "星の砂", "錆びた歯車", "輝く聖杯", "黒い薔薇",
    "白い孔雀", "秘められた林檎", "古の地図", "時の歯車", "深海の真珠", "暁の鳥", "黄昏の雫", "不滅の炎", "氷の心臓", "奏でる竪琴",
    "空の器", "叡智の梟", "秘密の鍵穴", "約束の指輪", "流れる雲", "不動の岩", "踊る影", "導きの杖", "遮られた眼", "語らぬ仮面"
]

PREFECTURE_LONGITUDES = {
    "北海道": 141.35,
    "青森": 140.74, "青森県": 140.74,
    "岩手": 141.15, "岩手県": 141.15,
    "宮城": 140.87, "宮城県": 140.87,
    "秋田": 140.10, "秋田県": 140.10,
    "山形": 140.36, "山形県": 140.36,
    "福島": 140.47, "福島県": 140.47,
    "茨城": 140.45, "茨城県": 140.45,
    "栃木": 139.88, "栃木県": 139.88,
    "群馬": 139.06, "群馬県": 139.06,
    "埼玉": 139.65, "埼玉県": 139.65,
    "千葉": 140.12, "千葉県": 140.12,
    "東京": 139.69, "東京都": 139.69,
    "神奈川": 139.64, "神奈川県": 139.64,
    "新潟": 139.02, "新潟県": 139.02,
    "富山": 137.21, "富山県": 137.21,
    "石川": 136.65, "石川県": 136.65,
    "福井": 136.22, "福井県": 136.22,
    "山梨": 138.57, "山梨県": 138.57,
    "長野": 138.18, "長野県": 138.18,
    "岐阜": 136.76, "岐阜県": 136.76,
    "静岡": 138.38, "静岡県": 138.38,
    "愛知": 136.91, "愛知県": 136.91,
    "三重": 136.51, "三重県": 136.51,
    "滋賀": 135.87, "滋賀県": 135.87,
    "京都": 135.77, "京都府": 135.77,
    "大阪": 135.50, "大阪府": 135.50,
    "兵庫": 135.18, "兵庫県": 135.18,
    "奈良": 135.83, "奈良県": 135.83,
    "和歌山": 135.17, "和歌山県": 135.17,
    "鳥取": 134.24, "鳥取県": 134.24,
    "島根": 132.75, "島根県": 132.75,
    "岡山": 133.93, "岡山県": 133.93,
    "広島": 132.46, "広島県": 132.46,
    "山口": 131.47, "山口県": 131.47,
    "徳島": 134.56, "徳島県": 134.56,
    "香川": 134.04, "香川県": 134.04,
    "愛媛": 132.77, "愛媛県": 132.77,
    "高知": 133.53, "高知県": 133.53,
    "福岡": 130.42, "福岡県": 130.42,
    "佐賀": 130.30, "佐賀県": 130.30,
    "長崎": 129.88, "長崎県": 129.88,
    "熊本": 130.74, "熊本県": 130.74,
    "大分": 131.61, "大分県": 131.61,
    "宮崎": 131.42, "宮崎県": 131.42,
    "鹿児島": 130.56, "鹿児島県": 130.56,
    "沖縄": 127.68, "沖縄県": 127.68,
}

FLOW_KIND_ALIASES = {
    "本日": "today",
    "今日": "today",
    "today": "today",
    "今週": "week",
    "week": "week",
    "今月": "month",
    "month": "month",
    "半年": "halfyear",
    "halfyear": "halfyear",
    "一年": "year",
    "1年": "year",
    "year": "year",
    "別の想い": "other",
    "別のこと": "other",
    "other": "other",
    "終える": "end",
    "終了": "end",
    "end": "end",
}


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"環境変数 {name} が未設定です")
    return value


LINE_CHANNEL_ACCESS_TOKEN = require_env("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = require_env("LINE_CHANNEL_SECRET")
OPENAI_API_KEY = require_env("OPENAI_API_KEY")
FIREBASE_SERVICE_ACCOUNT_JSON = require_env("FIREBASE_SERVICE_ACCOUNT_JSON")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

if not firebase_admin._apps:
    key_dict = json.loads(FIREBASE_SERVICE_ACCOUNT_JSON)
    firebase_admin.initialize_app(credentials.Certificate(key_dict))

db = firestore.client()
oracle_engine = OracleEngine(
    openai_client=openai_client,
    model_name=OPENAI_MODEL,
)

_user_locks: Dict[str, threading.Lock] = {}
_user_locks_guard = threading.Lock()


def get_user_lock(user_id: str) -> threading.Lock:
    with _user_locks_guard:
        if user_id not in _user_locks:
            _user_locks[user_id] = threading.Lock()
        return _user_locks[user_id]


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", (text or "").strip())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def trim_history(text: Optional[str], max_chars: int = MAX_CHAT_HISTORY_CHARS) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def parse_postback_data(data: str) -> Dict[str, str]:
    try:
        return dict(parse_qsl(data, keep_blank_values=True))
    except Exception:
        logger.warning("postback parse failed: %s", data)
        return {}


def normalize_yes_no(text: str) -> Optional[str]:
    t = normalize_text(text).lower()
    yes_set = {"はい", "うん", "ok", "okay", "yes", "y", "確定", "これでいい", "良い", "よい", "大丈夫"}
    no_set = {"いいえ", "修正", "やり直す", "違う", "ちがう", "no", "n"}
    if t in yes_set:
        return "yes"
    if t in no_set:
        return "no"
    return None


def extract_clean_name(text: str) -> str:
    t = normalize_text(text)
    t = re.sub(r"(です|と申します|だよ|と申す|だ|といいます|って呼びます|って呼んで)$", "", t).strip()
    t = re.sub(r"\s+", " ", t)
    return t[:30] if t else ""


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def detect_prefecture_longitude(text: str) -> Optional[float]:
    t = normalize_text(text)
    for key, lon in PREFECTURE_LONGITUDES.items():
        if key in t:
            return lon
    return None


def detect_prefecture_label(text: str) -> Optional[str]:
    t = normalize_text(text)
    for key in PREFECTURE_LONGITUDES.keys():
        if key in t:
            if key.endswith("県") or key.endswith("都") or key.endswith("府") or key == "北海道":
                return key
            return f"{key}県"
    return None


def build_user_profile(user_data: Dict[str, Any]) -> Dict[str, Any]:
    profile: Dict[str, Any] = {
        "name": user_data.get("name", PROFILE_DEFAULT_NAME)
    }

    birth_date = user_data.get("birth_date")
    if birth_date:
        try:
            y, m, d = map(int, birth_date.split("-"))
            profile.update({
                "birth_year": y,
                "birth_month": m,
                "birth_day": d,
            })
        except Exception:
            logger.warning("invalid birth_date format: %s", birth_date)

    birth_hour = safe_int(user_data.get("birth_hour"), None)
    birth_minute = safe_int(user_data.get("birth_minute"), DEFAULT_UNKNOWN_MINUTE)
    birth_second = safe_int(user_data.get("birth_second"), DEFAULT_UNKNOWN_SECOND)

    if user_data.get("birth_longitude") is not None:
        birth_longitude = safe_float(user_data.get("birth_longitude"), DEFAULT_BIRTH_LONGITUDE)
    else:
        birth_longitude = safe_float(user_data.get("birth_place_longitude"), DEFAULT_BIRTH_LONGITUDE)

    if birth_hour is not None:
        profile["birth_hour"] = birth_hour
    if birth_minute is not None:
        profile["birth_minute"] = birth_minute
    if birth_second is not None:
        profile["birth_second"] = birth_second
    if birth_longitude is not None:
        profile["birth_longitude"] = birth_longitude

    return profile


def push_text(user_id: str, text: str, quick_reply: Optional[QuickReply] = None) -> bool:
    try:
        line_bot_api.push_message(user_id, TextSendMessage(text=text, quick_reply=quick_reply))
        return True
    except LineBotApiError:
        logger.exception("LINE push failed user_id=%s", user_id)
        return False
    except Exception:
        logger.exception("Unexpected LINE push failure user_id=%s", user_id)
        return False


def push_template(user_id: str, template_message: TemplateSendMessage) -> bool:
    try:
        line_bot_api.push_message(user_id, template_message)
        return True
    except LineBotApiError:
        logger.exception("LINE template push failed user_id=%s", user_id)
        return False
    except Exception:
        logger.exception("Unexpected LINE template push failure user_id=%s", user_id)
        return False


def get_payment_guide_text(user_data: Dict[str, Any]) -> str:
    name = user_data.get("name", PROFILE_DEFAULT_NAME)
    return (
        f"{name}様は、初回の詠歌までは受け取れています。\n"
        "本日・今週・今月の流れをさらに読むには、『深読みの扉』をお開きください。\n\n"
        f"導きの頁: {PAYMENT_URL}\n\n"
        "開いた後に『決済完了』と送ってください。"
    )


def get_premium_guide_text(user_data: Dict[str, Any]) -> str:
    name = user_data.get("name", PROFILE_DEFAULT_NAME)
    return (
        f"{name}様はいま『深読みの扉』の内にあります。\n"
        "半年・一年の深い流れを読むには、『深奥の扉』をお開きください。\n\n"
        f"導きの頁: {PREMIUM_PAYMENT_URL}\n\n"
        "開いた後に『深奥完了』と送ってください。"
    )


def send_initial_greeting(user_id: str) -> bool:
    return push_text(
        user_id,
        "私は“識”。天伝詔より賜った詠歌を、あなたに伝える存在です。今回はどの様な想いをお持ちになりましたか？"
    )


def send_birthday_picker(user_id: str, message: str) -> bool:
    template = ButtonsTemplate(
        text=message,
        actions=[
            DatetimePickerAction(
                label="暦で選ぶ",
                data="action=set_birthday",
                mode="date",
                initial="1995-01-01",
            )
        ],
    )
    return push_template(
        user_id,
        TemplateSendMessage(alt_text="生まれた日を選ぶ", template=template),
    )


def send_time_picker(user_id: str) -> bool:
    template = ButtonsTemplate(
        text="生まれた時刻は分かりますか？",
        actions=[
            DatetimePickerAction(
                label="時刻を選ぶ",
                data="action=set_birthtime",
                mode="time",
                initial="12:00",
            ),
            PostbackAction(
                label="分からない",
                data="action=set_birthtime_unknown",
                display_text="分からない",
            ),
        ],
    )
    return push_template(
        user_id,
        TemplateSendMessage(alt_text="生まれた時刻を選ぶ", template=template),
    )


def send_birth_prefecture_prompt(user_id: str) -> bool:
    return push_text(
        user_id,
        "最後に、出生地の都道府県を教えてください。\n"
        "分からない場合は『不明』、海外なら『海外』でも大丈夫です。"
    )


def send_profile_confirm(user_id: str, date_str: str, time_str: str, prefecture_str: str) -> bool:
    text = (
        f"生年月日: {date_str}\n"
        f"出生時刻: {time_str}\n"
        f"出生地: {prefecture_str}\n\n"
        "こちらの刻印でよろしいでしょうか？"
    )
    items = [
        QuickReplyButton(
            action=PostbackAction(
                label="はい",
                data="action=confirm_profile&res=yes",
                display_text="はい",
            )
        ),
        QuickReplyButton(
            action=PostbackAction(
                label="やり直す",
                data="action=confirm_profile&res=no",
                display_text="やり直す",
            )
        ),
    ]
    return push_text(user_id, text, quick_reply=QuickReply(items=items))


def send_restart_confirm(user_id: str) -> bool:
    items = [
        QuickReplyButton(
            action=PostbackAction(
                label="はい",
                data="action=restart&res=yes",
                display_text="はい",
            )
        ),
        QuickReplyButton(
            action=PostbackAction(
                label="いいえ",
                data="action=restart&res=no",
                display_text="いいえ",
            )
        ),
    ]
    return push_text(user_id, "新しい想いについて、あらためて詠歌を読みますか？", quick_reply=QuickReply(items=items))


def send_motif_picker(user_id: str) -> List[str]:
    sampled = random.sample(ALL_MOTIFS, 4)
    items = [
        QuickReplyButton(
            action=PostbackAction(
                label=m,
                data=f"action=select_motif&label={m}",
                display_text=m,
            )
        )
        for m in sampled
    ]
    push_text(
        user_id,
        "準備は整いました。心に触れる象徴を一つ選んでください。",
        quick_reply=QuickReply(items=items),
    )
    return sampled


def send_coldread_options(user_id: str) -> bool:
    items = [
        QuickReplyButton(
            action=PostbackAction(
                label="ありました",
                data="action=coldread_reply&value=yes",
                display_text="ありました",
            )
        ),
        QuickReplyButton(
            action=PostbackAction(
                label="少しあります",
                data="action=coldread_reply&value=partial",
                display_text="少しあります",
            )
        ),
        QuickReplyButton(
            action=PostbackAction(
                label="どちらとも言えない",
                data="action=coldread_reply&value=unclear",
                display_text="どちらとも言えない",
            )
        ),
        QuickReplyButton(
            action=PostbackAction(
                label="まだ分かりません",
                data="action=coldread_reply&value=unknown",
                display_text="まだ分かりません",
            )
        ),
    ]
    return push_text(
        user_id,
        "いまの詠歌との響き方に近いものを選んでください。",
        quick_reply=QuickReply(items=items),
    )


def send_followup_menu(user_id: str, user_data: Dict[str, Any]) -> bool:
    consult = user_data.get("last_consult_label", "この想い")
    buttons = [
        QuickReplyButton(
            action=PostbackAction(label="本日", data="action=followup_menu&kind=today", display_text="本日")
        ),
        QuickReplyButton(
            action=PostbackAction(label="今週", data="action=followup_menu&kind=week", display_text="今週")
        ),
        QuickReplyButton(
            action=PostbackAction(label="今月", data="action=followup_menu&kind=month", display_text="今月")
        ),
        QuickReplyButton(
            action=PostbackAction(label="半年", data="action=followup_menu&kind=halfyear", display_text="半年")
        ),
        QuickReplyButton(
            action=PostbackAction(label="一年", data="action=followup_menu&kind=year", display_text="一年")
        ),
        QuickReplyButton(
            action=PostbackAction(label="別の想いを読む", data="action=followup_menu&kind=other", display_text="別の想いを読む")
        ),
        QuickReplyButton(
            action=PostbackAction(label="ここで終える", data="action=followup_menu&kind=end", display_text="ここで終える")
        ),
    ]

    return push_text(
        user_id,
        f"どの流れを読みたいですか？\n今の想い: {consult}について",
        quick_reply=QuickReply(items=buttons[:13]),
    )


def get_user_ref(user_id: str):
    return db.collection("users").document(user_id)


def get_sessions_ref(user_id: str):
    return get_user_ref(user_id).collection("sessions")


def get_session_ref(user_id: str, session_id: str):
    return get_sessions_ref(user_id).document(session_id)


def get_messages_ref(user_id: str, session_id: str):
    return get_session_ref(user_id, session_id).collection("messages")


def load_user(user_id: str) -> Dict[str, Any]:
    snap = get_user_ref(user_id).get()
    data = snap.to_dict() or {}
    data["__exists__"] = snap.exists

    if "plan_status" not in data:
        data["plan_status"] = PLAN_FREE
    if "free_sessions_remaining" not in data:
        data["free_sessions_remaining"] = DEFAULT_FREE_SESSIONS

    if not data.get("phase"):
        if not data.get("pending_consult") and not data.get("name"):
            data["phase"] = PHASE_WAIT_INITIAL_CONSULT
        elif not data.get("name"):
            data["phase"] = PHASE_WAIT_NAME
        elif not data.get("birth_date"):
            data["phase"] = PHASE_WAIT_BIRTH_DATE
        elif data.get("birth_hour") is None:
            data["phase"] = PHASE_WAIT_BIRTH_TIME
        elif not data.get("birth_prefecture") and data.get("birth_place_unknown") is not True:
            data["phase"] = PHASE_WAIT_BIRTH_PREFECTURE
        elif not data.get("is_profile_confirmed"):
            data["phase"] = PHASE_WAIT_PROFILE_CONFIRM
        elif data.get("is_dialogue_mode"):
            data["phase"] = PHASE_DIALOGUE
        else:
            data["phase"] = PHASE_WAIT_RESTART_CONFIRM

    return data


def save_user(user_id: str, patch: Dict[str, Any]) -> None:
    patch["updated_at"] = now_iso()
    get_user_ref(user_id).set(patch, merge=True)


def delete_user_fields(user_id: str, field_names: List[str]) -> None:
    patch = {name: DELETE_FIELD for name in field_names}
    patch["updated_at"] = now_iso()
    get_user_ref(user_id).set(patch, merge=True)


def reset_user(user_id: str) -> None:
    get_user_ref(user_id).delete()


def finalize_profile_confirm_text(user_data: Dict[str, Any]) -> str:
    hour = safe_int(user_data.get("birth_hour"), None)
    minute = safe_int(user_data.get("birth_minute"), 0)

    if hour is None:
        return "未設定"

    if int(hour) == DEFAULT_UNKNOWN_HOUR and user_data.get("birth_time_unknown") is True:
        return "不明（正午として計算）"

    return f"{int(hour):02d}:{int(minute):02d}頃"


def finalize_prefecture_confirm_text(user_data: Dict[str, Any]) -> str:
    if user_data.get("birth_prefecture"):
        return str(user_data.get("birth_prefecture"))
    if user_data.get("birth_place_unknown"):
        return "不明（代表経度で計算）"
    return "未設定"


def log_usage_if_any(result: Dict[str, Any], user_id: str) -> None:
    try:
        summary = result.get("summary", {}) or {}
        usage = summary.get("usage_metadata")
        if usage:
            logger.info("usage user_id=%s usage=%s", user_id, usage)
    except Exception:
        logger.exception("usage log failed")


def can_start_initial_reading(user_data: Dict[str, Any]) -> bool:
    if user_data.get("plan_status") in {PLAN_PAID, PLAN_PREMIUM}:
        return True
    return int(user_data.get("free_sessions_remaining", DEFAULT_FREE_SESSIONS)) > 0


def consume_session_credit_if_needed(user_id: str, user_data: Dict[str, Any]) -> None:
    if user_data.get("plan_status") in {PLAN_PAID, PLAN_PREMIUM}:
        return
    remaining = int(user_data.get("free_sessions_remaining", DEFAULT_FREE_SESSIONS))
    if remaining > 0:
        save_user(user_id, {"free_sessions_remaining": remaining - 1})


def create_new_session(user_id: str, user_data: Dict[str, Any], consult_text: str, motif_label: str) -> str:
    session_id = str(uuid4())
    session_doc = {
        "session_id": session_id,
        "user_name": user_data.get("name", PROFILE_DEFAULT_NAME),
        "status": "active",
        "plan_status": user_data.get("plan_status", PLAN_FREE),
        "motif_label": motif_label,
        "initial_consult": consult_text,
        "started_at": now_iso(),
        "updated_at": now_iso(),
    }
    get_session_ref(user_id, session_id).set(session_doc, merge=True)
    save_user(user_id, {"current_session_id": session_id})
    return session_id


def close_session(user_id: str, session_id: str) -> None:
    get_session_ref(user_id, session_id).set(
        {"status": "closed", "closed_at": now_iso(), "updated_at": now_iso()},
        merge=True,
    )


def append_session_message(
    user_id: str,
    session_id: str,
    role: str,
    text: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    message_id = str(uuid4())
    payload = {
        "message_id": message_id,
        "role": role,
        "text": text,
        "created_at": now_iso(),
    }
    if extra:
        payload.update(extra)

    get_messages_ref(user_id, session_id).document(message_id).set(payload, merge=True)
    get_session_ref(user_id, session_id).set({"updated_at": now_iso()}, merge=True)


def build_followup_prompt(kind: str, user_data: Dict[str, Any]) -> str:
    consult = user_data.get("last_consult_text", "今回の想い")
    if kind == "today":
        return f"今の想い『{consult}』に関して、本日一日の感情・対人・判断の流れを、占いとしてやさしく読んでください。"
    if kind == "week":
        return f"今の想い『{consult}』に関して、次の月曜日から始まる一週間の流れを、転機と気配を中心に占いとして読んでください。"
    if kind == "month":
        return f"今の想い『{consult}』に関して、今月の流れを、心の揺れ・対人・動きやすい時期を含めて占いとして読んでください。"
    if kind == "halfyear":
        return f"今の想い『{consult}』に関して、これから半年の流れを、転換点や整い方を中心に占いとして読んでください。"
    if kind == "year":
        return f"今の想い『{consult}』に関して、これから一年の流れを、節目や運の巡りを中心に占いとして読んでください。"
    return f"今の想い『{consult}』について、やさしく流れを読んでください。"


def normalize_flow_kind(text: str) -> Optional[str]:
    t = normalize_text(text)
    return FLOW_KIND_ALIASES.get(t)


def build_consult_label(text: str) -> str:
    t = normalize_text(text)
    if len(t) <= 18:
        return t
    return t[:18] + "…"


def build_coldread_ack(value: str) -> str:
    if value == "yes":
        return "やはり、その揺れはすでに水面に現れていたのですね。では次に、どの流れを読みましょうか。"
    if value == "partial":
        return "まだ輪郭は淡いものの、すでに兆しは触れ始めているようです。では次に、どの流れを読みましょうか。"
    if value == "unclear":
        return "まだ形が定まりきっていないのかもしれません。曖昧なままでも大丈夫です。では次に、どの流れを読みましょうか。"
    return "いま無理に答えを決めなくても大丈夫です。流れから先に読むこともできます。では次に、どの流れを読みましょうか。"


def is_same_consult_repeated(user_data: Dict[str, Any], consult_text: str) -> bool:
    last_text = normalize_text(user_data.get("last_consult_text", ""))
    new_text = normalize_text(consult_text)
    return bool(last_text and new_text and last_text == new_text)


def process_and_push_reply(
    user_id: str,
    user_text: str,
    motif_label: Optional[str] = None,
    selected_date: Optional[str] = None,
    selected_time: Optional[str] = None,
    followup_kind: Optional[str] = None,
    coldread_value: Optional[str] = None,
) -> None:
    lock = get_user_lock(user_id)

    with lock:
        try:
            user_data = load_user(user_id)
            text = normalize_text(user_text)
            phase = user_data.get("phase", PHASE_WAIT_INITIAL_CONSULT)

            logger.info(
                "process start user_id=%s phase=%s text=%s motif=%s date=%s time=%s followup=%s coldread=%s",
                user_id, phase, text, motif_label, selected_date, selected_time, followup_kind, coldread_value
            )

            if text == "リセット":
                keep_plan = user_data.get("plan_status", PLAN_FREE)
                keep_free = int(user_data.get("free_sessions_remaining", DEFAULT_FREE_SESSIONS))
                reset_user(user_id)
                save_user(
                    user_id,
                    {
                        "phase": PHASE_WAIT_INITIAL_CONSULT,
                        "plan_status": keep_plan,
                        "free_sessions_remaining": keep_free,
                    },
                )
                send_initial_greeting(user_id)
                return

            if not user_data.get("__exists__"):
                save_user(
                    user_id,
                    {
                        "phase": PHASE_WAIT_INITIAL_CONSULT,
                        "plan_status": PLAN_FREE,
                        "free_sessions_remaining": DEFAULT_FREE_SESSIONS,
                    },
                )
                send_initial_greeting(user_id)
                return

            if text == "決済完了":
                save_user(user_id, {"plan_status": PLAN_PAID, "phase": PHASE_WAIT_FOLLOWUP_MENU})
                push_text(user_id, "『深読みの扉』が開かれました。では、どの流れを読みましょうか。")
                send_followup_menu(user_id, load_user(user_id))
                return

            if text == "深奥完了":
                save_user(user_id, {"plan_status": PLAN_PREMIUM, "phase": PHASE_WAIT_FOLLOWUP_MENU})
                push_text(user_id, "『深奥の扉』が開かれました。では、より長い巡りまで読んでいきましょう。")
                send_followup_menu(user_id, load_user(user_id))
                return

            if phase == PHASE_WAIT_INITIAL_CONSULT:
                if not text:
                    send_initial_greeting(user_id)
                    return

                save_user(
                    user_id,
                    {
                        "pending_consult": text,
                        "pending_consult_label": build_consult_label(text),
                        "phase": PHASE_WAIT_NAME,
                    },
                )
                push_text(
                    user_id,
                    "想いは受け取りました。\n"
                    "まず、あなたを何とお呼びすればよいでしょうか？\n"
                    "お呼びするお名前だけを送ってください。"
                )
                return

            if phase == PHASE_WAIT_NAME:
                clean_name = extract_clean_name(text)
                if not clean_name:
                    push_text(user_id, "お呼びするお名前だけを、短く送ってください。")
                    return

                save_user(
                    user_id,
                    {
                        "name": clean_name,
                        "phase": PHASE_WAIT_BIRTH_DATE,
                        "is_profile_confirmed": False,
                        "birth_time_unknown": False,
                        "birth_place_unknown": False,
                        "plan_status": user_data.get("plan_status", PLAN_FREE),
                        "free_sessions_remaining": int(user_data.get("free_sessions_remaining", DEFAULT_FREE_SESSIONS)),
                    },
                )
                send_birthday_picker(
                    user_id,
                    f"……{clean_name}様ですね。次に、生まれた日を教えてください。"
                )
                return

            if phase == PHASE_WAIT_BIRTH_DATE:
                if selected_date:
                    save_user(user_id, {"birth_date": selected_date, "phase": PHASE_WAIT_BIRTH_TIME})
                    send_time_picker(user_id)
                    return

                send_birthday_picker(
                    user_id,
                    f"{user_data.get('name', PROFILE_DEFAULT_NAME)}様。生まれた日を教えてください。"
                )
                return

            if phase == PHASE_WAIT_BIRTH_TIME:
                birth_date = user_data.get("birth_date")
                if not birth_date:
                    save_user(user_id, {"phase": PHASE_WAIT_BIRTH_DATE})
                    send_birthday_picker(user_id, "先に生まれた日を教えてください。")
                    return

                if selected_time:
                    try:
                        hh_str, mm_str = selected_time.split(":")
                        hour = int(hh_str)
                        minute = int(mm_str)
                    except Exception:
                        push_text(user_id, "時刻の読み取りに失敗しました。もう一度お試しください。")
                        send_time_picker(user_id)
                        return

                    save_user(
                        user_id,
                        {
                            "birth_hour": hour,
                            "birth_minute": minute,
                            "birth_second": 0,
                            "birth_time_unknown": False,
                            "phase": PHASE_WAIT_BIRTH_PREFECTURE,
                        },
                    )
                    send_birth_prefecture_prompt(user_id)
                    return

                if text == "UNKNOWN_TIME":
                    save_user(
                        user_id,
                        {
                            "birth_hour": DEFAULT_UNKNOWN_HOUR,
                            "birth_minute": DEFAULT_UNKNOWN_MINUTE,
                            "birth_second": DEFAULT_UNKNOWN_SECOND,
                            "birth_time_unknown": True,
                            "phase": PHASE_WAIT_BIRTH_PREFECTURE,
                        },
                    )
                    send_birth_prefecture_prompt(user_id)
                    return

                send_time_picker(user_id)
                return

            if phase == PHASE_WAIT_BIRTH_PREFECTURE:
                t = normalize_text(text)
                if not t:
                    send_birth_prefecture_prompt(user_id)
                    return

                if t in {"不明", "わからない", "分からない", "海外"}:
                    save_user(
                        user_id,
                        {
                            "birth_prefecture": "不明",
                            "birth_place_unknown": True,
                            "birth_longitude": DEFAULT_BIRTH_LONGITUDE,
                            "phase": PHASE_WAIT_PROFILE_CONFIRM,
                        },
                    )
                    latest = load_user(user_id)
                    send_profile_confirm(
                        user_id,
                        latest.get("birth_date", "未設定"),
                        finalize_profile_confirm_text(latest),
                        "不明（代表経度で計算）",
                    )
                    return

                detected_lon = detect_prefecture_longitude(t)
                detected_pref = detect_prefecture_label(t)
                if detected_lon is None:
                    push_text(
                        user_id,
                        "都道府県名で送ってください。\n"
                        "分からない場合は『不明』でも大丈夫です。"
                    )
                    return

                save_user(
                    user_id,
                    {
                        "birth_prefecture": detected_pref or t,
                        "birth_place_unknown": False,
                        "birth_longitude": detected_lon,
                        "phase": PHASE_WAIT_PROFILE_CONFIRM,
                    },
                )
                latest = load_user(user_id)
                send_profile_confirm(
                    user_id,
                    latest.get("birth_date", "未設定"),
                    finalize_profile_confirm_text(latest),
                    detected_pref or t,
                )
                return

            if phase == PHASE_WAIT_PROFILE_CONFIRM:
                if text == "CONFIRM_YES":
                    yn = "yes"
                elif text == "CONFIRM_NO":
                    yn = "no"
                else:
                    yn = normalize_yes_no(text)

                if yn == "yes":
                    save_user(user_id, {"is_profile_confirmed": True, "phase": PHASE_WAIT_MOTIF})
                    sampled = send_motif_picker(user_id)
                    save_user(user_id, {"last_presented_motifs": sampled})
                    return

                if yn == "no":
                    save_user(user_id, {"is_profile_confirmed": False, "phase": PHASE_WAIT_BIRTH_DATE})
                    delete_user_fields(
                        user_id,
                        [
                            "birth_date", "birth_hour", "birth_minute", "birth_second",
                            "birth_time_unknown", "birth_prefecture", "birth_place_unknown", "birth_longitude"
                        ]
                    )
                    send_birthday_picker(user_id, "承知しました。では、もう一度生まれた日を教えてください。")
                    return

                send_profile_confirm(
                    user_id,
                    user_data.get("birth_date", "未設定"),
                    finalize_profile_confirm_text(user_data),
                    finalize_prefecture_confirm_text(user_data),
                )
                return

            if phase == PHASE_WAIT_RESTART_CONFIRM:
                if text == "RESTART_YES":
                    yn = "yes"
                elif text == "RESTART_NO":
                    yn = "no"
                else:
                    yn = normalize_yes_no(text)

                if yn is None and text:
                    if is_same_consult_repeated(user_data, text):
                        push_text(
                            user_id,
                            "前とまったく同じ問いをすぐに重ねると、像が揺れてしまいます。\n"
                            "少し角度を変えるか、別の想いを置いてみてください。"
                        )
                        return

                    save_user(user_id, {"temp_restart_text": text})
                    send_restart_confirm(user_id)
                    return

                if yn == "no":
                    delete_user_fields(user_id, ["temp_restart_text"])
                    push_text(user_id, "承知しました。私はまた静かな縁にてお待ちしております。")
                    return

                if yn == "yes":
                    consult_seed = user_data.get("temp_restart_text", "")
                    consult_seed = normalize_text(consult_seed)

                    if not consult_seed:
                        push_text(user_id, "今回はどの様な想いをお持ちになりましたか？")
                        return

                    if len(consult_seed) <= 15:
                        save_user(
                            user_id,
                            {
                                "temp_category": consult_seed,
                                "phase": PHASE_WAIT_CONSULT_DETAIL,
                            },
                        )
                        push_text(
                            user_id,
                            f"『{consult_seed}』についてですね。\n"
                            "いま胸の内にある背景や、最近の空気感をもう少しだけ教えてください。"
                        )
                        return

                    save_user(
                        user_id,
                        {
                            "pending_consult": consult_seed,
                            "pending_consult_label": build_consult_label(consult_seed),
                            "temp_restart_text": DELETE_FIELD,
                            "temp_category": DELETE_FIELD,
                            "phase": PHASE_WAIT_MOTIF,
                        },
                    )
                    sampled = send_motif_picker(user_id)
                    save_user(user_id, {"last_presented_motifs": sampled})
                    return

            if phase == PHASE_WAIT_PAYMENT:
                push_text(user_id, get_payment_guide_text(user_data))
                return

            if phase == PHASE_WAIT_CONSULT_DETAIL:
                category = user_data.get("temp_category", "いまの想い")
                detail = text if text else "まだ言葉にはなり切っていない"
                combined_consult = f"{category}\n背景: {detail}"

                save_user(
                    user_id,
                    {
                        "pending_consult": combined_consult,
                        "pending_consult_label": build_consult_label(category),
                        "temp_category": DELETE_FIELD,
                        "temp_restart_text": DELETE_FIELD,
                        "phase": PHASE_WAIT_MOTIF,
                    },
                )
                sampled = send_motif_picker(user_id)
                save_user(user_id, {"last_presented_motifs": sampled})
                return

            if phase == PHASE_WAIT_MOTIF:
                if not motif_label:
                    sampled = user_data.get("last_presented_motifs")
                    if not sampled:
                        sampled = send_motif_picker(user_id)
                        save_user(user_id, {"last_presented_motifs": sampled})
                    else:
                        items = [
                            QuickReplyButton(
                                action=PostbackAction(
                                    label=m,
                                    data=f"action=select_motif&label={m}",
                                    display_text=m,
                                )
                            )
                            for m in sampled
                        ]
                        push_text(
                            user_id,
                            "心に触れる象徴を一つ選んでください。",
                            quick_reply=QuickReply(items=items),
                        )
                    return

                if not can_start_initial_reading(user_data):
                    save_user(user_id, {"phase": PHASE_WAIT_PAYMENT})
                    push_text(user_id, get_payment_guide_text(user_data))
                    return

                profile = build_user_profile(user_data)
                required_keys = ["birth_year", "birth_month", "birth_day"]
                missing = [k for k in required_keys if k not in profile]
                if missing:
                    logger.warning("profile missing keys user_id=%s missing=%s profile=%s", user_id, missing, profile)
                    push_text(user_id, "刻印に不足があるようです。もう一度、最初から整えてください。")
                    return

                consult_text = user_data.get("pending_consult", "これからの運勢")

                if is_same_consult_repeated(user_data, consult_text):
                    push_text(
                        user_id,
                        "前と同じ問いを続けて読むと、水面が濁ります。\n"
                        "少し別の角度から問い直してみてください。"
                    )
                    return

                logger.info(
                    "oracle start user_id=%s motif=%s consult=%s profile=%s",
                    user_id, motif_label, consult_text[:120], profile
                )

                push_text(user_id, "想いは届きました。天伝詔が詠歌を読んでいます。")
                consume_session_credit_if_needed(user_id, user_data)

                result = oracle_engine.predict(
                    user_profile=profile,
                    user_text=consult_text,
                    motif_label=motif_label,
                    is_dialogue=False,
                    chat_history="",
                )
                reply_text = result["message"]
                log_usage_if_any(result, user_id)

                history = f"識の詠歌: {reply_text}\n"
                history = trim_history(history)

                session_id = create_new_session(user_id, user_data, consult_text, motif_label)
                append_session_message(
                    user_id,
                    session_id,
                    "oracle",
                    reply_text,
                    extra={"summary": result.get("summary", {})},
                )

                save_user(
                    user_id,
                    {
                        "phase": PHASE_WAIT_COLDREAD_RESPONSE,
                        "is_dialogue_mode": False,
                        "chat_history": history,
                        "last_motif": motif_label,
                        "last_oracle_message": reply_text,
                        "last_oracle_summary": result.get("summary", {}),
                        "last_consult_text": consult_text,
                        "last_consult_label": user_data.get("pending_consult_label", build_consult_label(consult_text)),
                        "pending_consult": DELETE_FIELD,
                        "pending_consult_label": DELETE_FIELD,
                        "temp_category": DELETE_FIELD,
                        "temp_restart_text": DELETE_FIELD,
                    },
                )
                push_text(user_id, reply_text)
                send_coldread_options(user_id)
                return

            if phase == PHASE_WAIT_COLDREAD_RESPONSE:
                value = coldread_value or normalize_text(text)

                if value in {"ありました", "yes"}:
                    normalized_value = "yes"
                elif value in {"少しあります", "partial"}:
                    normalized_value = "partial"
                elif value in {"どちらとも言えない", "unclear"}:
                    normalized_value = "unclear"
                else:
                    normalized_value = "unknown"

                session_id = user_data.get("current_session_id")
                if session_id:
                    append_session_message(
                        user_id,
                        session_id,
                        "user",
                        f"照合の答え: {normalized_value}",
                    )

                history = trim_history(user_data.get("chat_history", "") + f"照合の答え: {normalized_value}\n")
                save_user(
                    user_id,
                    {
                        "phase": PHASE_WAIT_FOLLOWUP_MENU,
                        "last_coldread_reply": normalized_value,
                        "chat_history": history,
                    },
                )
                push_text(user_id, build_coldread_ack(normalized_value))
                send_followup_menu(user_id, load_user(user_id))
                return

            if phase == PHASE_WAIT_FOLLOWUP_MENU:
                kind = followup_kind or normalize_flow_kind(text) or text

                if kind == "end":
                    session_id = user_data.get("current_session_id")
                    if session_id:
                        close_session(user_id, session_id)

                    save_user(
                        user_id,
                        {
                            "phase": PHASE_WAIT_RESTART_CONFIRM,
                            "is_dialogue_mode": False,
                            "current_session_id": DELETE_FIELD,
                            "chat_history": DELETE_FIELD,
                        },
                    )
                    push_text(user_id, "では、ここで灯りを静かに閉じます。また必要なときに声をかけてください。")
                    return

                if kind == "other":
                    save_user(user_id, {"phase": PHASE_WAIT_RESTART_CONFIRM})
                    push_text(user_id, "新しく読みたい想いを教えてください。")
                    return

                plan = user_data.get("plan_status", PLAN_FREE)
                if kind in {"today", "week", "month"} and plan == PLAN_FREE:
                    push_text(user_id, get_payment_guide_text(user_data))
                    return
                if kind in {"halfyear", "year"} and plan != PLAN_PREMIUM:
                    if plan == PLAN_PAID:
                        push_text(user_id, get_premium_guide_text(user_data))
                    else:
                        push_text(user_id, get_payment_guide_text(user_data))
                    return

                if kind not in {"today", "week", "month", "halfyear", "year"}:
                    send_followup_menu(user_id, user_data)
                    return

                profile = build_user_profile(user_data)
                history = trim_history(user_data.get("chat_history", ""))
                session_id = user_data.get("current_session_id")

                prompt_text = build_followup_prompt(kind, user_data)
                result = oracle_engine.predict(
                    user_profile=profile,
                    user_text=prompt_text,
                    motif_label=user_data.get("last_motif", "静かなる光"),
                    is_dialogue=True,
                    chat_history=history,
                )
                reply_text = result["message"]
                log_usage_if_any(result, user_id)

                if session_id:
                    append_session_message(
                        user_id,
                        session_id,
                        "oracle",
                        reply_text,
                        extra={"summary": result.get("summary", {}), "followup_kind": kind},
                    )

                new_history = trim_history(history + f"識: {reply_text}\n")
                save_user(
                    user_id,
                    {
                        "phase": PHASE_WAIT_FOLLOWUP_MENU,
                        "chat_history": new_history,
                        "last_oracle_message": reply_text,
                        "last_oracle_summary": result.get("summary", {}),
                    },
                )
                push_text(user_id, reply_text)
                send_followup_menu(user_id, load_user(user_id))
                return

            if phase == PHASE_DIALOGUE:
                profile = build_user_profile(user_data)
                history = trim_history(user_data.get("chat_history", ""))
                session_id = user_data.get("current_session_id")

                if session_id:
                    append_session_message(user_id, session_id, "user", text)

                result = oracle_engine.predict(
                    user_profile=profile,
                    user_text=text,
                    motif_label=user_data.get("last_motif", "静かなる光"),
                    is_dialogue=True,
                    chat_history=history,
                )
                reply_text = result["message"]
                log_usage_if_any(result, user_id)

                if session_id:
                    append_session_message(
                        user_id,
                        session_id,
                        "oracle",
                        reply_text,
                        extra={"summary": result.get("summary", {})},
                    )

                user_name = user_data.get("name", PROFILE_DEFAULT_NAME)
                new_history = history + f"{user_name}: {text}\n識: {reply_text}\n"
                save_user(user_id, {"chat_history": trim_history(new_history)})
                push_text(user_id, reply_text)
                return

            logger.warning("unknown phase user_id=%s phase=%s", user_id, phase)
            save_user(user_id, {"phase": PHASE_WAIT_RESTART_CONFIRM})
            push_text(user_id, "少し視界が揺らぎました。もう一度、いま読みたい想いを教えてください。")

        except LineBotApiError:
            logger.exception("LINE API error while processing reply for user_id=%s", user_id)
            return

        except Exception:
            logger.exception("Error while processing reply for user_id=%s", user_id)
            try:
                ok = push_text(user_id, "識の視界が揺らぎました。もう一度だけ、同じ内容を送ってみてください。")
                if not ok:
                    logger.error("Fallback push also failed for user_id=%s", user_id)
            except Exception:
                logger.exception("Failed to push fallback message for user_id=%s", user_id)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok", "model": OPENAI_MODEL, "service": SERVICE_NAME}


@app.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature")
    if not signature:
        raise HTTPException(status_code=400, detail="Missing X-Line-Signature")

    body = await request.body()
    body_text = body.decode("utf-8")

    try:
        handler.handle(body_text, signature)
    except InvalidSignatureError:
        logger.warning("Invalid signature")
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception:
        logger.exception("Callback handling failed")
        raise HTTPException(status_code=500, detail="Callback error")

    return JSONResponse({"ok": True})


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event: MessageEvent):
    user_id = event.source.user_id
    text = normalize_text(event.message.text)

    thread = threading.Thread(
        target=process_and_push_reply,
        args=(user_id, text),
        daemon=True,
    )
    thread.start()


@handler.add(PostbackEvent)
def handle_postback(event: PostbackEvent):
    user_id = event.source.user_id
    query = parse_postback_data(event.postback.data)
    action = query.get("action")

    if action == "confirm_profile":
        text_val = "CONFIRM_YES" if query.get("res") == "yes" else "CONFIRM_NO"
        threading.Thread(target=process_and_push_reply, args=(user_id, text_val), daemon=True).start()
        return

    if action == "restart":
        text_val = "RESTART_YES" if query.get("res") == "yes" else "RESTART_NO"
        threading.Thread(target=process_and_push_reply, args=(user_id, text_val), daemon=True).start()
        return

    if action == "select_motif":
        threading.Thread(target=process_and_push_reply, args=(user_id, "", query.get("label")), daemon=True).start()
        return

    if action == "set_birthday":
        selected_date = event.postback.params.get("date") if event.postback.params else None
        threading.Thread(target=process_and_push_reply, args=(user_id, "", None, selected_date), daemon=True).start()
        return

    if action == "set_birthtime":
        selected_time = event.postback.params.get("time") if event.postback.params else None
        threading.Thread(target=process_and_push_reply, args=(user_id, "", None, None, selected_time), daemon=True).start()
        return

    if action == "set_birthtime_unknown":
        threading.Thread(target=process_and_push_reply, args=(user_id, "UNKNOWN_TIME"), daemon=True).start()
        return

    if action == "coldread_reply":
        threading.Thread(
            target=process_and_push_reply,
            args=(user_id, "", None, None, None, None, query.get("value")),
            daemon=True,
        ).start()
        return

    if action == "followup_menu":
        kind = query.get("kind", "")
        threading.Thread(
            target=process_and_push_reply,
            args=(user_id, "", None, None, None, kind),
            daemon=True,
        ).start()
        return

    logger.warning("Unknown postback action: %s", action)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )
