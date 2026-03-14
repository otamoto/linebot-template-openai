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
from linebot.exceptions import InvalidSignatureError

from google import genai

import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin.firestore import DELETE_FIELD

from oracle_engine import OracleEngine


# =========================================================
# 基本設定
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s : %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="SHIKI LINE Bot")


# =========================================================
# 定数
# =========================================================
PHASE_WAIT_NAME = "waiting_name"
PHASE_WAIT_BIRTH_DATE = "waiting_birth_date"
PHASE_WAIT_BIRTH_TIME = "waiting_birth_time"
PHASE_WAIT_PROFILE_CONFIRM = "waiting_profile_confirm"
PHASE_WAIT_RESTART_CONFIRM = "waiting_restart_confirm"
PHASE_WAIT_CONSULT_DETAIL = "waiting_consult_detail"
PHASE_WAIT_MOTIF = "waiting_motif"
PHASE_DIALOGUE = "dialogue"

DEFAULT_UNKNOWN_HOUR = 12
DEFAULT_UNKNOWN_MINUTE = 0
DEFAULT_BIRTH_LONGITUDE = float(os.getenv("DEFAULT_BIRTH_LONGITUDE", "135.0"))

MAX_CHAT_HISTORY_CHARS = 5000
PROFILE_DEFAULT_NAME = "あなた"

ALL_MOTIFS = [
    "銀の鍵", "砂時計", "古びた鏡", "聖なる滴", "琥珀の蝶", "折れた剣", "青い月", "羅針盤", "封じられた手紙", "金の天秤",
    "揺れる灯火", "水晶の髑髏", "沈黙の鐘", "茨の冠", "無垢な羽根", "双子の蛇", "星の砂", "錆びた歯車", "輝く聖杯", "黒い薔薇",
    "白い孔雀", "秘められた林檎", "古の地図", "時の歯車", "深海の真珠", "暁の鳥", "黄昏の雫", "不滅の炎", "氷の心臓", "奏でる竪琴",
    "空の器", "叡智の梟", "秘密の鍵穴", "約束の指輪", "流れる雲", "不動の岩", "踊る影", "導きの杖", "遮られた眼", "語らぬ仮面"
]


# =========================================================
# 環境変数
# =========================================================
def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"環境変数 {name} が未設定です")
    return value


LINE_CHANNEL_ACCESS_TOKEN = require_env("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = require_env("LINE_CHANNEL_SECRET")
GEMINI_API_KEY = require_env("GEMINI_API_KEY")
FIREBASE_SERVICE_ACCOUNT_JSON = require_env("FIREBASE_SERVICE_ACCOUNT_JSON")
CHAT_MODEL = os.getenv("CHAT_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))


# =========================================================
# 外部サービス初期化
# =========================================================
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)
genai_client = genai.Client(api_key=GEMINI_API_KEY)

if not firebase_admin._apps:
    try:
        key_dict = json.loads(FIREBASE_SERVICE_ACCOUNT_JSON)
    except json.JSONDecodeError as e:
        raise RuntimeError("FIREBASE_SERVICE_ACCOUNT_JSON が正しいJSONではありません") from e
    firebase_admin.initialize_app(credentials.Certificate(key_dict))

db = firestore.client()
oracle_engine = OracleEngine(gemini_client=genai_client, model_name=CHAT_MODEL)


# =========================================================
# ユーザー単位ロック
# =========================================================
_user_locks: Dict[str, threading.Lock] = {}
_user_locks_guard = threading.Lock()


def get_user_lock(user_id: str) -> threading.Lock:
    with _user_locks_guard:
        if user_id not in _user_locks:
            _user_locks[user_id] = threading.Lock()
        return _user_locks[user_id]


# =========================================================
# ユーティリティ
# =========================================================
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
    yes_set = {"はい", "うん", "ok", "okay", "yes", "y", "確定", "これでいい", "良い", "よい"}
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
    birth_second = safe_int(user_data.get("birth_second"), 0)
    birth_longitude = safe_float(user_data.get("birth_longitude"), DEFAULT_BIRTH_LONGITUDE)

    if birth_hour is not None:
        profile["birth_hour"] = birth_hour
    if birth_minute is not None:
        profile["birth_minute"] = birth_minute
    if birth_second is not None:
        profile["birth_second"] = birth_second
    if birth_longitude is not None:
        profile["birth_longitude"] = birth_longitude

    return profile


def push_text(user_id: str, text: str, quick_reply: Optional[QuickReply] = None) -> None:
    line_bot_api.push_message(user_id, TextSendMessage(text=text, quick_reply=quick_reply))


def send_birthday_picker(user_id: str, message: str) -> None:
    template = ButtonsTemplate(
        text=message,
        actions=[
            DatetimePickerAction(
                label="カレンダーで選択",
                data="action=set_birthday",
                mode="date",
                initial="1995-01-01",
            )
        ],
    )
    line_bot_api.push_message(
        user_id,
        TemplateSendMessage(alt_text="誕生日選択", template=template),
    )


def send_time_picker(user_id: str) -> None:
    template = ButtonsTemplate(
        text="生まれた時刻は分かりますか？",
        actions=[
            DatetimePickerAction(
                label="時刻を選択",
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
    line_bot_api.push_message(
        user_id,
        TemplateSendMessage(alt_text="時刻選択", template=template),
    )


def send_profile_confirm(user_id: str, date_str: str, time_str: str) -> None:
    text = f"生年月日: {date_str}\n出生時刻: {time_str}\n\nこちらの刻印でよろしいでしょうか？"
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
    push_text(user_id, text, quick_reply=QuickReply(items=items))


def send_restart_confirm(user_id: str) -> None:
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
    push_text(user_id, "新たなる観測を始めますか？", quick_reply=QuickReply(items=items))


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


def get_user_ref(user_id: str):
    return db.collection("users").document(user_id)


def load_user(user_id: str) -> Dict[str, Any]:
    snap = get_user_ref(user_id).get()
    data = snap.to_dict() or {}
    if not data.get("phase"):
        if not data.get("name"):
            data["phase"] = PHASE_WAIT_NAME
        elif not data.get("birth_date"):
            data["phase"] = PHASE_WAIT_BIRTH_DATE
        elif data.get("birth_hour") is None:
            data["phase"] = PHASE_WAIT_BIRTH_TIME
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


# =========================================================
# 本体ロジック
# =========================================================
def process_and_push_reply(
    user_id: str,
    user_text: str,
    motif_label: Optional[str] = None,
    selected_date: Optional[str] = None,
    selected_time: Optional[str] = None,
) -> None:
    lock = get_user_lock(user_id)

    with lock:
        try:
            user_data = load_user(user_id)

            text = normalize_text(user_text)
            phase = user_data.get("phase", PHASE_WAIT_NAME)

            logger.info(
                "process start user_id=%s phase=%s text=%s motif=%s date=%s time=%s",
                user_id,
                phase,
                text,
                motif_label,
                selected_date,
                selected_time,
            )

            # -------------------------------------------------
            # 共通: リセット
            # -------------------------------------------------
            if text == "リセット":
                reset_user(user_id)
                push_text(
                    user_id,
                    "ようこそ、探究者の方。新たな観測を始めましょう。\n"
                    "まずは、あなた様をどのようにお呼びすればよろしいですか？"
                    "（『〇〇です』などは付けず、お呼びするお名前のみを送信してください）"
                )
                return

            # -------------------------------------------------
            # phase: 名前待ち
            # -------------------------------------------------
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
                    },
                )
                send_birthday_picker(
                    user_id,
                    f"……{clean_name}様ですね。心に刻みました。次に、あなたの生まれた日を教えてください。"
                )
                return

            # -------------------------------------------------
            # phase: 生年月日待ち
            # -------------------------------------------------
            if phase == PHASE_WAIT_BIRTH_DATE:
                if selected_date:
                    save_user(
                        user_id,
                        {
                            "birth_date": selected_date,
                            "phase": PHASE_WAIT_BIRTH_TIME,
                        },
                    )
                    send_time_picker(user_id)
                    return

                send_birthday_picker(
                    user_id,
                    f"{user_data.get('name', PROFILE_DEFAULT_NAME)}様。観測を始める前に、生まれた日を教えてください。"
                )
                return

            # -------------------------------------------------
            # phase: 出生時刻待ち
            # -------------------------------------------------
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
                            "birth_longitude": user_data.get("birth_longitude", DEFAULT_BIRTH_LONGITUDE),
                            "birth_time_unknown": False,
                            "phase": PHASE_WAIT_PROFILE_CONFIRM,
                        },
                    )
                    send_profile_confirm(user_id, birth_date, f"{hour:02d}:{minute:02d}頃")
                    return

                if text == "UNKNOWN_TIME":
                    save_user(
                        user_id,
                        {
                            "birth_hour": DEFAULT_UNKNOWN_HOUR,
                            "birth_minute": DEFAULT_UNKNOWN_MINUTE,
                            "birth_second": 0,
                            "birth_longitude": user_data.get("birth_longitude", DEFAULT_BIRTH_LONGITUDE),
                            "birth_time_unknown": True,
                            "phase": PHASE_WAIT_PROFILE_CONFIRM,
                        },
                    )
                    send_profile_confirm(user_id, birth_date, "不明（正午として計算）")
                    return

                send_time_picker(user_id)
                return

            # -------------------------------------------------
            # phase: プロフィール確認待ち
            # -------------------------------------------------
            if phase == PHASE_WAIT_PROFILE_CONFIRM:
                if text == "CONFIRM_YES":
                    yn = "yes"
                elif text == "CONFIRM_NO":
                    yn = "no"
                else:
                    yn = normalize_yes_no(text)

                if yn == "yes":
                    save_user(
                        user_id,
                        {
                            "is_profile_confirmed": True,
                            "phase": PHASE_WAIT_RESTART_CONFIRM,
                        },
                    )
                    push_text(
                        user_id,
                        f"刻印が完成しました。今、{user_data.get('name', PROFILE_DEFAULT_NAME)}様が一番視たいことは何でしょうか。"
                    )
                    return

                if yn == "no":
                    save_user(
                        user_id,
                        {
                            "is_profile_confirmed": False,
                            "phase": PHASE_WAIT_BIRTH_DATE,
                        },
                    )
                    delete_user_fields(
                        user_id,
                        ["birth_date", "birth_hour", "birth_minute", "birth_second", "birth_time_unknown"]
                    )
                    send_birthday_picker(
                        user_id,
                        "承知いたしました。では、もう一度生まれた日を正しく教えてください。"
                    )
                    return

                send_profile_confirm(
                    user_id,
                    user_data.get("birth_date", "未設定"),
                    finalize_profile_confirm_text(user_data),
                )
                return

            # -------------------------------------------------
            # phase: 新規観測開始確認
            # -------------------------------------------------
            if phase == PHASE_WAIT_RESTART_CONFIRM:
                if text == "RESTART_YES":
                    yn = "yes"
                elif text == "RESTART_NO":
                    yn = "no"
                else:
                    yn = normalize_yes_no(text)

                if yn is None and text:
                    save_user(
                        user_id,
                        {
                            "temp_restart_text": text,
                        },
                    )
                    send_restart_confirm(user_id)
                    return

                if yn == "no":
                    delete_user_fields(user_id, ["temp_restart_text"])
                    push_text(user_id, "承知いたしました。私はまた淵にてお待ちしております。")
                    return

                if yn == "yes":
                    consult_seed = user_data.get("temp_restart_text", "これからの運勢")
                    consult_seed = normalize_text(consult_seed)

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
                            f"……「{consult_seed}」についてですね。その奥にある想いを、もう少しだけ詳しく教えていただけますか？\n"
                            "（具体的な状況や、今感じている不安などを教えていただけると、より深く観測できます）"
                        )
                        return

                    save_user(
                        user_id,
                        {
                            "pending_consult": consult_seed,
                            "temp_restart_text": DELETE_FIELD,
                            "temp_category": DELETE_FIELD,
                            "phase": PHASE_WAIT_MOTIF,
                        },
                    )
                    sampled = send_motif_picker(user_id)
                    save_user(user_id, {"last_presented_motifs": sampled})
                    return

            # -------------------------------------------------
            # phase: 詳細相談待ち
            # -------------------------------------------------
            if phase == PHASE_WAIT_CONSULT_DETAIL:
                category = user_data.get("temp_category", "これからの運勢")
                detail = text if text else "詳しい事情はまだ言葉にならない"
                combined_consult = f"{category}（詳細：{detail}）"

                save_user(
                    user_id,
                    {
                        "pending_consult": combined_consult,
                        "temp_category": DELETE_FIELD,
                        "temp_restart_text": DELETE_FIELD,
                        "phase": PHASE_WAIT_MOTIF,
                    },
                )
                sampled = send_motif_picker(user_id)
                save_user(user_id, {"last_presented_motifs": sampled})
                return

            # -------------------------------------------------
            # phase: モチーフ待ち
            # -------------------------------------------------
            if phase == PHASE_WAIT_MOTIF:
                if not motif_label:
                    sampled = user_data.get("last_presented_motifs")
                    if not sampled or len(sampled) < 1:
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

                profile = build_user_profile(user_data)
                consult_text = user_data.get("pending_consult", "これからの運勢")

                result = oracle_engine.predict(
                    user_profile=profile,
                    user_text=consult_text,
                    motif_label=motif_label,
                    is_dialogue=False,
                    chat_history="",
                )
                reply_text = result["message"]

                history = f"識の神託: {reply_text}\n"
                history = trim_history(history)

                save_user(
                    user_id,
                    {
                        "is_dialogue_mode": True,
                        "phase": PHASE_DIALOGUE,
                        "chat_history": history,
                        "last_motif": motif_label,
                        "last_oracle_summary": result.get("summary", {}),
                        "pending_consult": DELETE_FIELD,
                        "temp_category": DELETE_FIELD,
                        "temp_restart_text": DELETE_FIELD,
                    },
                )
                push_text(user_id, reply_text)
                return

            # -------------------------------------------------
            # phase: 対話モード
            # -------------------------------------------------
            if phase == PHASE_DIALOGUE:
                profile = build_user_profile(user_data)
                history = trim_history(user_data.get("chat_history", ""))

                result = oracle_engine.predict(
                    user_profile=profile,
                    user_text=text,
                    motif_label=user_data.get("last_motif", "静かなる光"),
                    is_dialogue=True,
                    chat_history=history,
                )
                reply_text = result["message"]

                if "[END_SESSION]" in reply_text:
                    reply_text = reply_text.replace("[END_SESSION]", "").strip()
                    save_user(
                        user_id,
                        {
                            "is_dialogue_mode": False,
                            "phase": PHASE_WAIT_RESTART_CONFIRM,
                            "last_oracle_summary": result.get("summary", {}),
                            "chat_history": DELETE_FIELD,
                            "pending_consult": DELETE_FIELD,
                            "temp_category": DELETE_FIELD,
                            "temp_restart_text": DELETE_FIELD,
                        },
                    )
                    push_text(user_id, reply_text)
                    return

                user_name = user_data.get("name", PROFILE_DEFAULT_NAME)
                new_history = history + f"{user_name}: {text}\n識: {reply_text}\n"
                new_history = trim_history(new_history)

                save_user(
                    user_id,
                    {
                        "chat_history": new_history,
                        "last_oracle_summary": result.get("summary", {}),
                    },
                )
                push_text(user_id, reply_text)
                return

            # -------------------------------------------------
            # 不明phaseの保険
            # -------------------------------------------------
            logger.warning("unknown phase user_id=%s phase=%s", user_id, phase)
            save_user(user_id, {"phase": PHASE_WAIT_RESTART_CONFIRM})
            push_text(user_id, "少し視界が揺らぎました。もう一度、今視たいことを教えてください。")
            return

        except Exception:
            logger.exception("Error while processing reply for user_id=%s", user_id)
            try:
                push_text(user_id, "識の視界が揺らぎました。もう一度だけ、同じ内容を送ってみてください。")
            except Exception:
                logger.exception("Failed to push fallback message for user_id=%s", user_id)


# =========================================================
# FastAPI / LINE callback
# =========================================================
@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok", "model": CHAT_MODEL}


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
        threading.Thread(
            target=process_and_push_reply,
            args=(user_id, text_val),
            daemon=True,
        ).start()
        return

    if action == "restart":
        text_val = "RESTART_YES" if query.get("res") == "yes" else "RESTART_NO"
        threading.Thread(
            target=process_and_push_reply,
            args=(user_id, text_val),
            daemon=True,
        ).start()
        return

    if action == "select_motif":
        threading.Thread(
            target=process_and_push_reply,
            args=(user_id, "", query.get("label")),
            daemon=True,
        ).start()
        return

    if action == "set_birthday":
        selected_date = None
        if event.postback.params:
            selected_date = event.postback.params.get("date")
        threading.Thread(
            target=process_and_push_reply,
            args=(user_id, "", None, selected_date),
            daemon=True,
        ).start()
        return

    if action == "set_birthtime":
        selected_time = None
        if event.postback.params:
            selected_time = event.postback.params.get("time")
        threading.Thread(
            target=process_and_push_reply,
            args=(user_id, "", None, None, selected_time),
            daemon=True,
        ).start()
        return

    if action == "set_birthtime_unknown":
        threading.Thread(
            target=process_and_push_reply,
            args=(user_id, "UNKNOWN_TIME"),
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
