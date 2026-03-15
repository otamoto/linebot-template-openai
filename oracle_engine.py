import os
import math
import hashlib
import logging
import re
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)


@dataclass
class PillarResult:
    year_pillar: str
    month_pillar: str
    day_pillar: str
    hour_pillar: Optional[str]
    year_hidden_stems: List[str]
    month_hidden_stems: List[str]
    day_hidden_stems: List[str]
    hour_hidden_stems: List[str]
    year_twelve_stage: str
    month_twelve_stage: str
    day_twelve_stage: str
    hour_twelve_stage: Optional[str]
    year_tsuhen: str
    month_tsuhen: str
    day_tsuhen: str
    hour_tsuhen: Optional[str]
    effective_year: int
    solar_longitude: float
    nine_star_year: str
    five_element_scores: Dict[str, float]
    self_strength_hint: str


class PreciseCalendar:
    JUKKAN = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
    JUNISHI = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]

    STEM_ELEMENT = {
        "甲": "木", "乙": "木",
        "丙": "火", "丁": "火",
        "戊": "土", "己": "土",
        "庚": "金", "辛": "金",
        "壬": "水", "癸": "水",
    }

    STEM_YINYANG = {
        "甲": "陽", "乙": "陰",
        "丙": "陽", "丁": "陰",
        "戊": "陽", "己": "陰",
        "庚": "陽", "辛": "陰",
        "壬": "陽", "癸": "陰",
    }

    BRANCH_ELEMENT_MAIN = {
        "子": "水", "丑": "土", "寅": "木", "卯": "木",
        "辰": "土", "巳": "火", "午": "火", "未": "土",
        "申": "金", "酉": "金", "戌": "土", "亥": "水",
    }

    HIDDEN_STEMS = {
        "子": ["癸"],
        "丑": ["己", "癸", "辛"],
        "寅": ["甲", "丙", "戊"],
        "卯": ["乙"],
        "辰": ["戊", "乙", "癸"],
        "巳": ["丙", "庚", "戊"],
        "午": ["丁", "己"],
        "未": ["己", "丁", "乙"],
        "申": ["庚", "壬", "戊"],
        "酉": ["辛"],
        "戌": ["戊", "辛", "丁"],
        "亥": ["壬", "甲"],
    }

    MONTH_BRANCH_ELEMENT_WEIGHTS = {
        "寅": {"木": 1.8, "火": 1.0, "土": 0.6, "金": 0.4, "水": 0.8},
        "卯": {"木": 2.0, "火": 0.8, "土": 0.5, "金": 0.3, "水": 0.7},
        "辰": {"木": 0.8, "火": 0.7, "土": 1.8, "金": 0.5, "水": 0.8},
        "巳": {"木": 0.7, "火": 1.9, "土": 0.9, "金": 0.5, "水": 0.3},
        "午": {"木": 0.6, "火": 2.0, "土": 0.8, "金": 0.4, "水": 0.2},
        "未": {"木": 0.7, "火": 0.8, "土": 1.9, "金": 0.5, "水": 0.4},
        "申": {"木": 0.3, "火": 0.5, "土": 0.8, "金": 1.9, "水": 0.9},
        "酉": {"木": 0.2, "火": 0.4, "土": 0.7, "金": 2.0, "水": 0.8},
        "戌": {"木": 0.4, "火": 0.8, "土": 1.9, "金": 0.8, "水": 0.4},
        "亥": {"木": 0.9, "火": 0.3, "土": 0.5, "金": 0.7, "水": 1.9},
        "子": {"木": 0.8, "火": 0.2, "土": 0.4, "金": 0.8, "水": 2.0},
        "丑": {"木": 0.5, "火": 0.4, "土": 1.8, "金": 0.8, "水": 0.9},
    }

    KIGAKU_LIST = [
        "一白水星", "二黒土星", "三碧木星", "四緑木星", "五黄土星",
        "六白金星", "七赤金星", "八白土星", "九紫火星"
    ]

    SEXAGENARY_DAY_BASE_JDN = 2445733

    TWELVE_STAGE_START = {
        "甲": "亥", "乙": "午",
        "丙": "寅", "丁": "酉",
        "戊": "寅", "己": "酉",
        "庚": "巳", "辛": "子",
        "壬": "申", "癸": "卯",
    }

    TWELVE_STAGES_FORWARD = [
        "長生", "沐浴", "冠帯", "建禄", "帝旺", "衰", "病", "死", "墓", "絶", "胎", "養"
    ]

    @staticmethod
    def julian_day(
        year: int,
        month: int,
        day: int,
        hour: int = 12,
        minute: int = 0,
        second: int = 0,
        longitude: float = 135.0,
    ) -> float:
        y, m = (year - 1, month + 12) if month <= 2 else (year, month)
        a = y // 100
        b = 2 - a + (a // 4)
        local_hour = hour + minute / 60.0 + second / 3600.0
        corrected_hour = local_hour + (longitude - 135.0) * (4.0 / 60.0)
        return (
            math.floor(365.25 * (y + 4716))
            + math.floor(30.6001 * (m + 1))
            + day
            + (corrected_hour / 24.0)
            + b
            - 1524.5
        )

    @staticmethod
    def solar_longitude(jd: float) -> float:
        t = (jd - 2451545.0) / 36525.0
        l0 = (280.46646 + 36000.76983 * t + 0.0003032 * t * t) % 360.0
        m = (357.52911 + 35999.05029 * t - 0.0001537 * t * t + (t ** 3) / 24490000.0) % 360.0
        mrad = math.radians(m)
        c = (
            (1.914602 - 0.004817 * t - 0.000014 * t * t) * math.sin(mrad)
            + (0.019993 - 0.000101 * t) * math.sin(2 * mrad)
            + 0.000289 * math.sin(3 * mrad)
        )
        true_long = l0 + c
        omega = 125.04 - 1934.136 * t
        lambda_app = true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))
        return lambda_app % 360.0

    @classmethod
    def effective_year_by_setsu(cls, birth_year: int, solar_long: float) -> int:
        return birth_year if solar_long >= 315.0 else birth_year - 1

    @classmethod
    def year_pillar(cls, effective_year: int) -> str:
        return cls.JUKKAN[(effective_year - 4) % 10] + cls.JUNISHI[(effective_year - 4) % 12]

    @classmethod
    def month_index_from_solar_longitude(cls, solar_long: float) -> int:
        return int(((solar_long - 315.0) % 360.0) // 30.0)

    @classmethod
    def month_pillar(cls, year_stem: str, solar_long: float) -> str:
        month_idx = cls.month_index_from_solar_longitude(solar_long)
        start_kan_map = {
            "甲": 2, "己": 2,
            "乙": 4, "庚": 4,
            "丙": 6, "辛": 6,
            "丁": 8, "壬": 8,
            "戊": 0, "癸": 0,
        }
        stem = cls.JUKKAN[(start_kan_map[year_stem] + month_idx) % 10]
        branch = cls.JUNISHI[(2 + month_idx) % 12]
        return stem + branch

    @classmethod
    def day_pillar(cls, jd: float) -> str:
        di = (int(math.floor(jd + 0.5)) - cls.SEXAGENARY_DAY_BASE_JDN) % 60
        return cls.JUKKAN[di % 10] + cls.JUNISHI[di % 12]

    @classmethod
    def hour_pillar(cls, day_stem: str, birth_hour: Optional[int]) -> Optional[str]:
        if birth_hour is None:
            return None
        hour_branch_idx = ((birth_hour + 1) // 2) % 12
        start_map = {
            "甲": 0, "己": 0,
            "乙": 2, "庚": 2,
            "丙": 4, "辛": 4,
            "丁": 6, "壬": 6,
            "戊": 8, "癸": 8,
        }
        stem = cls.JUKKAN[(start_map[day_stem] + hour_branch_idx) % 10]
        branch = cls.JUNISHI[hour_branch_idx]
        return stem + branch

    @classmethod
    def nine_star_year(cls, effective_year: int) -> str:
        star_num = 11 - (effective_year % 9)
        while star_num > 9:
            star_num -= 9
        if star_num <= 0:
            star_num += 9
        return cls.KIGAKU_LIST[star_num - 1]

    @classmethod
    def get_hidden_stems(cls, pillar: Optional[str]) -> List[str]:
        if not pillar:
            return []
        branch = pillar[1]
        return cls.HIDDEN_STEMS.get(branch, [])

    @classmethod
    def _element_generates(cls, element: str) -> str:
        return {"木": "火", "火": "土", "土": "金", "金": "水", "水": "木"}[element]

    @classmethod
    def _element_controls(cls, element: str) -> str:
        return {"木": "土", "火": "金", "土": "水", "金": "木", "水": "火"}[element]

    @classmethod
    def get_tsuhen(cls, day_stem: str, target_stem: str) -> str:
        day_el = cls.STEM_ELEMENT[day_stem]
        day_yy = cls.STEM_YINYANG[day_stem]
        tgt_el = cls.STEM_ELEMENT[target_stem]
        tgt_yy = cls.STEM_YINYANG[target_stem]
        same_yy = day_yy == tgt_yy

        if day_el == tgt_el:
            return "比肩" if same_yy else "劫財"
        if cls._element_generates(day_el) == tgt_el:
            return "食神" if same_yy else "傷官"
        if cls._element_controls(day_el) == tgt_el:
            return "偏財" if same_yy else "正財"
        if cls._element_controls(tgt_el) == day_el:
            return "偏官" if same_yy else "正官"
        if cls._element_generates(tgt_el) == day_el:
            return "偏印" if same_yy else "印綬"
        return "不明"

    @classmethod
    def get_twelve_stage(cls, day_stem: str, branch: Optional[str]) -> Optional[str]:
        if not branch:
            return None
        start_branch = cls.TWELVE_STAGE_START[day_stem]
        start_idx = cls.JUNISHI.index(start_branch)
        target_idx = cls.JUNISHI.index(branch)
        is_yang = cls.STEM_YINYANG[day_stem] == "陽"
        diff = (target_idx - start_idx) % 12 if is_yang else (start_idx - target_idx) % 12
        return cls.TWELVE_STAGES_FORWARD[diff]

    @classmethod
    def compute_five_element_scores(
        cls,
        year_pillar: str,
        month_pillar: str,
        day_pillar: str,
        hour_pillar: Optional[str],
    ) -> Dict[str, float]:
        scores = {"木": 0.0, "火": 0.0, "土": 0.0, "金": 0.0, "水": 0.0}

        def add_stem(stem: str, weight: float):
            scores[cls.STEM_ELEMENT[stem]] += weight

        def add_hidden(branch: str, base_weights: List[float]):
            hs = cls.HIDDEN_STEMS.get(branch, [])
            for i, stem in enumerate(hs):
                w = base_weights[i] if i < len(base_weights) else 0.2
                scores[cls.STEM_ELEMENT[stem]] += w

        pillars = [year_pillar, month_pillar, day_pillar]
        if hour_pillar:
            pillars.append(hour_pillar)

        stem_weights = [1.0, 1.4, 1.2, 0.8] if hour_pillar else [1.0, 1.4, 1.2]
        hidden_weights = {
            0: [0.7, 0.25, 0.15],
            1: [1.2, 0.35, 0.2],
            2: [0.9, 0.3, 0.15],
            3: [0.6, 0.2, 0.1],
        }

        for idx, pillar in enumerate(pillars):
            stem = pillar[0]
            branch = pillar[1]
            add_stem(stem, stem_weights[idx])
            add_hidden(branch, hidden_weights[idx])

        month_branch = month_pillar[1]
        season_weights = cls.MONTH_BRANCH_ELEMENT_WEIGHTS.get(month_branch, {})
        for el, mul in season_weights.items():
            scores[el] *= mul

        return {k: round(v, 3) for k, v in scores.items()}

    @classmethod
    def evaluate_self_strength_hint(
        cls,
        day_stem: str,
        month_pillar: str,
        five_scores: Dict[str, float],
    ) -> str:
        dm_element = cls.STEM_ELEMENT[day_stem]
        resource_element = None
        for el, gen in {"木": "火", "火": "土", "土": "金", "金": "水", "水": "木"}.items():
            if gen == dm_element:
                resource_element = el
                break

        same_score = five_scores.get(dm_element, 0.0)
        resource_score = five_scores.get(resource_element, 0.0) if resource_element else 0.0
        support = same_score + resource_score
        total = sum(five_scores.values()) or 1.0
        ratio = support / total

        month_branch = month_pillar[1]
        season_main = cls.BRANCH_ELEMENT_MAIN[month_branch]

        if season_main == dm_element:
            ratio += 0.08
        elif resource_element and season_main == resource_element:
            ratio += 0.05

        if ratio >= 0.42:
            return "やや身強〜身強寄り"
        if ratio <= 0.25:
            return "やや身弱〜身弱寄り"
        return "中和寄り"


class BioRhythm:
    @staticmethod
    def _days_since_birth(
        birth_year: int,
        birth_month: int,
        birth_day: int,
        now_utc: Optional[datetime] = None,
    ) -> int:
        now_utc = now_utc or datetime.now(timezone.utc)
        birth = datetime(birth_year, birth_month, birth_day, tzinfo=timezone.utc)
        return (now_utc.date() - birth.date()).days

    @classmethod
    def build(cls, birth_year: int, birth_month: int, birth_day: int) -> Dict[str, Any]:
        days = cls._days_since_birth(birth_year, birth_month, birth_day)
        physical = math.sin(2 * math.pi * days / 23.0)
        emotional = math.sin(2 * math.pi * days / 28.0)
        intellectual = math.sin(2 * math.pi * days / 33.0)

        def label(v: float) -> str:
            if v >= 0.55:
                return "高い"
            if v >= 0.15:
                return "やや高い"
            if v > -0.15:
                return "揺らぎやすい"
            if v > -0.55:
                return "やや低い"
            return "低い"

        return {
            "physical": round(physical, 3),
            "emotional": round(emotional, 3),
            "intellectual": round(intellectual, 3),
            "physical_label": label(physical),
            "emotional_label": label(emotional),
            "intellectual_label": label(intellectual),
        }


class OracleEngine:
    TAROT_LIST = [
        "愚者", "魔術師", "女教皇", "女帝", "皇帝", "教皇", "恋人",
        "戦車", "力", "隠者", "運命の輪", "正義", "吊るされた男",
        "死神", "節制", "悪魔", "塔", "星", "月", "太陽", "審判", "世界"
    ]

    def __init__(self, openai_client, model_name: Optional[str] = None):
        self.openai_client = openai_client
        self.cal = PreciseCalendar()
        self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-5-mini")

    @staticmethod
    def _stable_seed(
        y: int,
        m: int,
        d: int,
        h: Optional[int],
        mn: int,
        motif_label: str,
        user_text: str,
    ) -> int:
        seed_str = f"{y:04d}-{m:02d}-{d:02d}|{h}|{mn}|{motif_label}|{user_text}"
        return int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest(), 16)

    @staticmethod
    def _extract_text_from_response(response: Any) -> str:
        text = getattr(response, "output_text", None)
        if text and str(text).strip():
            return str(text).strip()

        try:
            outputs = getattr(response, "output", None) or []
            chunks = []
            for item in outputs:
                content = getattr(item, "content", None) or []
                for c in content:
                    if getattr(c, "type", None) == "output_text":
                        t = getattr(c, "text", None)
                        if t:
                            chunks.append(t)
            merged = "\n".join(chunks).strip()
            if merged:
                return merged
        except Exception:
            pass

        return "……時の水面がまだ閉じています。"

    @staticmethod
    def _extract_usage_metadata(response: Any) -> Dict[str, Any]:
        try:
            usage = getattr(response, "usage", None)
            if usage is None:
                return {}
            out: Dict[str, Any] = {}
            for key in ["input_tokens", "output_tokens", "total_tokens"]:
                val = getattr(usage, key, None)
                if val is not None:
                    out[key] = val
            return out
        except Exception:
            return {}

    def classify_request_intent(self, user_text: str) -> Dict[str, str]:
        prompt = f"""
あなたは占いサービスの相談分類補助です。
次の相談文が、占いとして読める依頼か、読めない依頼かを判定してください。

# ルール
- 本人や相手の気持ち、未来傾向、相性、時期、選択の流れは読める
- 試験の合否を断定、失せ物の現在地特定、ギャンブルや当てもの、犯罪の助長は読めない
- ただし、犯罪被害、不安、怒り、恐怖、対人トラブルの相談は読める
- ただし、試験前の心の流れや努力の向きは読める
- ただし、失せ物そのものの場所は読めないが、探す向きや心の整え方は読める
- 相談者が感情表現として「殺してやりたい」などと言っているだけで、即犯罪依頼にしない
- 文脈重視で判定する
- 同じ問いを短時間でそのまま繰り返している場合は repeat とする

# 出力形式
JSONのみ
{{
  "allow": "yes" または "no",
  "reason": "normal/exam/lost_item/gambling/crime/repeat/other",
  "note": "20文字以内の短い補足"
}}

相談文:
{user_text}
""".strip()

        try:
            response = self.openai_client.responses.create(
                model=self.model_name,
                input=prompt,
            )
            raw = self._extract_text_from_response(response)
            data = json.loads(raw)
            allow = str(data.get("allow", "yes")).lower()
            reason = str(data.get("reason", "normal")).lower()
            note = str(data.get("note", "")).strip()
            return {
                "allow": "yes" if allow == "yes" else "no",
                "reason": reason,
                "note": note,
            }
        except Exception:
            logger.exception("classify_request_intent failed")
            return {
                "allow": "yes",
                "reason": "normal",
                "note": "",
            }

    @staticmethod
    def build_refusal_message(reason: str) -> str:
        if reason == "exam":
            return (
                "その問いは、合否そのものを断じる読みには向きません。\n"
                "ただ、試みの流れや、力を整えやすい時期なら読むことができます。"
            )
        if reason == "lost_item":
            return (
                "失せ物の在り処そのものを、識が正確に指し示すことはできません。\n"
                "ただ、探す向きや、見つかりやすい流れなら読むことができます。"
            )
        if reason == "gambling":
            return (
                "当てものや賭けの結果を狙って読むことは、識の役目の外にあります。\n"
                "ただ、運の使いどころや、いまの心の偏りなら読むことができます。"
            )
        if reason == "crime":
            return (
                "人を害することへ力を貸す読みは行えません。\n"
                "ただ、怒りや恐れの流れを静めるための読みならお手伝いできます。"
            )
        if reason == "repeat":
            return (
                "まったく同じ問いを重ねると、水面が揺れて像が曇ります。\n"
                "少し角度を変えた問いにしてみてください。"
            )
        return (
            "その問いは、いまの識には正しく像を結びにくいようです。\n"
            "別の角度からなら読めることがあります。"
        )

    def _build_prompt(
        self,
        user_name: str,
        motif_label: str,
        user_text: str,
        pillars: PillarResult,
        eki_num: int,
        tarot_name: str,
        biorhythm: Dict[str, Any],
        is_dialogue: bool,
        chat_history: str,
    ) -> str:
        five_str = " / ".join([f"{k}:{v}" for k, v in pillars.five_element_scores.items()])

        common_observation = f"""
# 観測断片
- 宿命: {pillars.year_pillar} / {pillars.month_pillar} / {pillars.day_pillar} / {pillars.hour_pillar or '時刻不明'}
- 蔵干:
  - 年: {", ".join(pillars.year_hidden_stems)}
  - 月: {", ".join(pillars.month_hidden_stems)}
  - 日: {", ".join(pillars.day_hidden_stems)}
  - 時: {", ".join(pillars.hour_hidden_stems) if pillars.hour_hidden_stems else 'なし'}
- 通変の気配:
  - 年干: {pillars.year_tsuhen}
  - 月干: {pillars.month_tsuhen}
  - 日干: {pillars.day_tsuhen}
  - 時干: {pillars.hour_tsuhen or 'なし'}
- 十二運:
  - 年支: {pillars.year_twelve_stage}
  - 月支: {pillars.month_twelve_stage}
  - 日支: {pillars.day_twelve_stage}
  - 時支: {pillars.hour_twelve_stage or 'なし'}
- 五行分布: {five_str}
- 日主の勢い: {pillars.self_strength_hint}
- 九星: {pillars.nine_star_year}
- 兆し: 象徴数 {eki_num} / 寓意の絵「{tarot_name}」
- 日々の波:
  - 身: {biorhythm["physical_label"]}
  - 情: {biorhythm["emotional_label"]}
  - 思: {biorhythm["intellectual_label"]}
- 問い: {user_text}
""".strip()

        if not is_dialogue:
            return f"""
あなたは「識」。
あなたは“天伝詔”より賜った詠歌を、人にわかる言葉へ解きほぐして伝える存在です。
{user_name}様が選んだ象徴「{motif_label}」に触れて、いま詠歌を受け取っています。

# 絶対条件
- これは占いであり、相談員・助言者・説教者ではない。
- 技法名は一切出さない。
- まず詠歌を、古い予言書や口伝のように、格調高く短く読む。
- 次に「--識の解読--」として、現代語でわかりやすく解説する。
- 初回出力では、本日・今週・今月・半年・一年の流れはまだ詳述しない。
- 実務指示、説教、命令口調はしない。
- 恐怖で縛らず、しかし軽くもしない。
- 断定しすぎず、傾向・流れ・兆しとして語る。
- 読みやすさを重視し、700文字前後以内に収める。
- 最後に、自然で柔らかい「照合の問い」を1つだけ置く。
- 照合の問いは、最近・過去・今現在に起こりがちなことを優しく確かめる聞き方にする。
- 照合の問いは、相手が「ありました / 少しあります / どちらとも言えない / まだ分かりません」で答えやすい内容にする。
- 詠歌本文にはカタカナと英単語を使わない。必ず和語・漢語中心で記す。
- 詠歌本文では外来語、横文字、現代技術語を使わない。
- 解読部分でも、不要な外来語は避ける。

# 出力形式
詠歌――
（詠歌本文）

--識の解読--
（現代語で3〜6文。相談者が理解しやすいように解説）

照合の問い――
（最近や過去に起きていそうなことを、柔らかく1問）

# 文体
- 詠歌部分は静かで神秘的
- 解読部分は優しい占い師の現代語
- 相談者の不安や迷いを見下さない
- “あなたを導く”ではなく、“流れを一緒に読む”感覚

{common_observation}
""".strip()

        return f"""
あなたは「識」。
あなたは“天伝詔”より賜った詠歌を、人の言葉へ解いて伝える占い師です。
{user_name}様との対話の続きとして応答してください。

# 絶対条件
- これは占いの続きであり、相談員・助言者・説教者ではない。
- 技法名は一切出さない。
- 実務指示、説教、命令口調を避ける。
- 回答は現代語で分かりやすく、優しい占い師の口調にする。
- 期間指定がある場合は、その期間の流れに集中して読む。
- 相談者本人、相手の気持ち、相性、選択や時期の流れは読める。
- 生死断定、犯罪の助長、当てもの、失せ物の場所特定、合否断定には寄らない。
- 不安を煽りすぎない。
- 最大420文字程度を目安にする。
- 利用者に見せる文では、世界観を壊す不要なカタカナ語・英語を避ける。
- 必要なら最後に一文だけ、やわらかい余韻や次の視点を添えてよい。
- 「最後に」「さて」「それでは」は使わない。

# 会話方針
- 期間の流れを具体と抽象の中間で読む
- 今日なら感情・対人・判断の気配を短く
- 今週・今月・半年・一年は、転機や巡りを中心に
- 相手が受け取りやすい言葉を選ぶ

# 会話履歴
{chat_history or "（履歴なし）"}

{common_observation}
""".strip()

    def build_four_pillars(
        self,
        birth_year: int,
        birth_month: int,
        birth_day: int,
        birth_hour: Optional[int] = None,
        birth_minute: int = 0,
        birth_second: int = 0,
        birth_longitude: float = 135.0,
    ) -> PillarResult:
        jd = self.cal.julian_day(
            birth_year,
            birth_month,
            birth_day,
            birth_hour if birth_hour is not None else 12,
            birth_minute,
            birth_second,
            birth_longitude,
        )
        sl = self.cal.solar_longitude(jd)
        ey = self.cal.effective_year_by_setsu(birth_year, sl)
        yp = self.cal.year_pillar(ey)
        mp = self.cal.month_pillar(yp[0], sl)
        dp = self.cal.day_pillar(jd)
        hp = self.cal.hour_pillar(dp[0], birth_hour)

        yh = self.cal.get_hidden_stems(yp)
        mh = self.cal.get_hidden_stems(mp)
        dh = self.cal.get_hidden_stems(dp)
        hh = self.cal.get_hidden_stems(hp)

        yt = self.cal.get_twelve_stage(dp[0], yp[1])
        mt = self.cal.get_twelve_stage(dp[0], mp[1])
        dt = self.cal.get_twelve_stage(dp[0], dp[1])
        ht = self.cal.get_twelve_stage(dp[0], hp[1]) if hp else None

        yts = self.cal.get_tsuhen(dp[0], yp[0])
        mts = self.cal.get_tsuhen(dp[0], mp[0])
        dts = self.cal.get_tsuhen(dp[0], dp[0])
        hts = self.cal.get_tsuhen(dp[0], hp[0]) if hp else None

        five_scores = self.cal.compute_five_element_scores(yp, mp, dp, hp)
        self_strength_hint = self.cal.evaluate_self_strength_hint(dp[0], mp, five_scores)

        return PillarResult(
            year_pillar=yp,
            month_pillar=mp,
            day_pillar=dp,
            hour_pillar=hp,
            year_hidden_stems=yh,
            month_hidden_stems=mh,
            day_hidden_stems=dh,
            hour_hidden_stems=hh,
            year_twelve_stage=yt,
            month_twelve_stage=mt,
            day_twelve_stage=dt,
            hour_twelve_stage=ht,
            year_tsuhen=yts,
            month_tsuhen=mts,
            day_tsuhen=dts,
            hour_tsuhen=hts,
            effective_year=ey,
            solar_longitude=sl,
            nine_star_year=self.cal.nine_star_year(ey),
            five_element_scores=five_scores,
            self_strength_hint=self_strength_hint,
        )

    def predict(
        self,
        user_profile: Dict[str, Any],
        user_text: str,
        motif_label: str,
        is_dialogue: bool = False,
        chat_history: str = "",
    ) -> Dict[str, Any]:
        try:
            classification = self.classify_request_intent(user_text)
            if classification.get("allow") != "yes":
                reason = classification.get("reason", "other")
                return {
                    "message": self.build_refusal_message(reason),
                    "summary": {
                        "forbidden_reason": reason,
                        "classification_note": classification.get("note", ""),
                        "model_name": self.model_name,
                    },
                    "topic": "refusal",
                }

            user_name = user_profile.get("name", "あなた")

            birth_year = int(user_profile["birth_year"])
            birth_month = int(user_profile["birth_month"])
            birth_day = int(user_profile["birth_day"])
            birth_hour = user_profile.get("birth_hour")
            birth_minute = int(user_profile.get("birth_minute", 0))
            birth_second = int(user_profile.get("birth_second", 0))
            birth_longitude = float(user_profile.get("birth_longitude", 135.0))

            if birth_hour is not None:
                birth_hour = int(birth_hour)

            pillars = self.build_four_pillars(
                birth_year=birth_year,
                birth_month=birth_month,
                birth_day=birth_day,
                birth_hour=birth_hour,
                birth_minute=birth_minute,
                birth_second=birth_second,
                birth_longitude=birth_longitude,
            )

            biorhythm = BioRhythm.build(
                birth_year=birth_year,
                birth_month=birth_month,
                birth_day=birth_day,
            )

            seed = self._stable_seed(
                birth_year,
                birth_month,
                birth_day,
                birth_hour,
                birth_minute,
                motif_label,
                user_text,
            )

            eki_num = (seed % 64) + 1
            tarot_name = self.TAROT_LIST[seed % len(self.TAROT_LIST)]

            prompt = self._build_prompt(
                user_name=user_name,
                motif_label=motif_label,
                user_text=user_text,
                pillars=pillars,
                eki_num=eki_num,
                tarot_name=tarot_name,
                biorhythm=biorhythm,
                is_dialogue=is_dialogue,
                chat_history=chat_history,
            )

            logger.info(
                "OracleEngine predict start model=%s user=%s motif=%s dialogue=%s",
                self.model_name, user_name, motif_label, is_dialogue
            )

            response = self.openai_client.responses.create(
                model=self.model_name,
                input=prompt,
            )

            message = self._extract_text_from_response(response)
            usage_metadata = self._extract_usage_metadata(response)

            logger.info(
                "OracleEngine predict success model=%s usage=%s",
                self.model_name, usage_metadata
            )

            return {
                "message": message,
                "summary": {
                    "pillars": {
                        "year": pillars.year_pillar,
                        "month": pillars.month_pillar,
                        "day": pillars.day_pillar,
                        "hour": pillars.hour_pillar,
                    },
                    "hidden_stems": {
                        "year": pillars.year_hidden_stems,
                        "month": pillars.month_hidden_stems,
                        "day": pillars.day_hidden_stems,
                        "hour": pillars.hour_hidden_stems,
                    },
                    "tsuhen": {
                        "year": pillars.year_tsuhen,
                        "month": pillars.month_tsuhen,
                        "day": pillars.day_tsuhen,
                        "hour": pillars.hour_tsuhen,
                    },
                    "twelve_stage": {
                        "year": pillars.year_twelve_stage,
                        "month": pillars.month_twelve_stage,
                        "day": pillars.day_twelve_stage,
                        "hour": pillars.hour_twelve_stage,
                    },
                    "five_element_scores": pillars.five_element_scores,
                    "self_strength_hint": pillars.self_strength_hint,
                    "nine_star_year": pillars.nine_star_year,
                    "effective_year": pillars.effective_year,
                    "solar_longitude": round(pillars.solar_longitude, 6),
                    "eki_num": eki_num,
                    "tarot_name": tarot_name,
                    "biorhythm": biorhythm,
                    "model_name": self.model_name,
                    "usage_metadata": usage_metadata,
                },
                "topic": "dialogue" if is_dialogue else "oracle",
            }

        except Exception as e:
            logger.exception("OracleEngine Error")
            msg = str(e)

            if "429" in msg or "rate limit" in msg.lower():
                return {
                    "message": "いま詠歌の窓辺に観測が集中しており、しばらく扉が閉じています。少し時間を置いて、もう一度声をかけてください。",
                    "summary": {},
                    "topic": "quota_error",
                }

            return {
                "message": f"識の視界が一時的に曇りました。({msg[:120]})",
                "summary": {},
                "topic": "error",
            }
