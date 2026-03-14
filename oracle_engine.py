import os
import math
import hashlib
import logging
from dataclasses import dataclass, asdict
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
    """
    強化版 東洋占術計算クラス

    実装範囲:
    - 年柱: 立春基準
    - 月柱: 節入り基準
    - 日柱
    - 時柱
    - 蔵干
    - 通変星
    - 十二運
    - 五行バランス
    - 九星年命
    """

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

    # 蔵干（簡略ではなく一般的な三層構造を採用）
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

    # 月令の力の簡易重み
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

    # 1984-02-02 を甲子日基準
    SEXAGENARY_DAY_BASE_JDN = 2445733

    # 十二運 起点表（各日主に対して長生となる支）
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
        """
        近似精度を少し高めた太陽黄経
        """
        T = (jd - 2451545.0) / 36525.0

        L0 = (
            280.46646
            + 36000.76983 * T
            + 0.0003032 * T * T
        ) % 360.0

        M = (
            357.52911
            + 35999.05029 * T
            - 0.0001537 * T * T
            + (T ** 3) / 24490000.0
        ) % 360.0

        Mrad = math.radians(M)
        C = (
            (1.914602 - 0.004817 * T - 0.000014 * T * T) * math.sin(Mrad)
            + (0.019993 - 0.000101 * T) * math.sin(2 * Mrad)
            + 0.000289 * math.sin(3 * Mrad)
        )

        true_long = L0 + C
        omega = 125.04 - 1934.136 * T
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
        """
        通変星（日干基準）
        """
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

        # 陽干は順行、陰干は逆行
        is_yang = cls.STEM_YINYANG[day_stem] == "陽"

        if is_yang:
            diff = (target_idx - start_idx) % 12
        else:
            diff = (start_idx - target_idx) % 12

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

        # 天干重み
        stem_weights = [1.0, 1.4, 1.2, 0.8] if hour_pillar else [1.0, 1.4, 1.2]
        # 蔵干重み
        hidden_weights = {
            0: [0.7, 0.25, 0.15],   # 年
            1: [1.2, 0.35, 0.2],    # 月
            2: [0.9, 0.3, 0.15],    # 日
            3: [0.6, 0.2, 0.1],     # 時
        }

        for idx, pillar in enumerate(pillars):
            stem = pillar[0]
            branch = pillar[1]
            add_stem(stem, stem_weights[idx])
            add_hidden(branch, hidden_weights[idx])

        # 月令補正
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
        """
        簡易 身強弱ヒント
        """
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

        # 月令一致なら少し補正
        if season_main == dm_element:
            ratio += 0.08
        elif resource_element and season_main == resource_element:
            ratio += 0.05

        if ratio >= 0.42:
            return "やや身強〜身強寄り"
        if ratio <= 0.25:
            return "やや身弱〜身弱寄り"
        return "中和寄り"

    @classmethod
    def build_four_pillars(
        cls,
        birth_year: int,
        birth_month: int,
        birth_day: int,
        birth_hour: Optional[int] = None,
        birth_minute: int = 0,
        birth_second: int = 0,
        birth_longitude: float = 135.0,
    ) -> PillarResult:
        jd = cls.julian_day(
            birth_year,
            birth_month,
            birth_day,
            birth_hour if birth_hour is not None else 12,
            birth_minute,
            birth_second,
            birth_longitude,
        )

        sl = cls.solar_longitude(jd)
        ey = cls.effective_year_by_setsu(birth_year, sl)
        yp = cls.year_pillar(ey)
        mp = cls.month_pillar(yp[0], sl)
        dp = cls.day_pillar(jd)
        hp = cls.hour_pillar(dp[0], birth_hour)

        yh = cls.get_hidden_stems(yp)
        mh = cls.get_hidden_stems(mp)
        dh = cls.get_hidden_stems(dp)
        hh = cls.get_hidden_stems(hp)

        yt = cls.get_twelve_stage(dp[0], yp[1])
        mt = cls.get_twelve_stage(dp[0], mp[1])
        dt = cls.get_twelve_stage(dp[0], dp[1])
        ht = cls.get_twelve_stage(dp[0], hp[1]) if hp else None

        yts = cls.get_tsuhen(dp[0], yp[0])
        mts = cls.get_tsuhen(dp[0], mp[0])
        dts = cls.get_tsuhen(dp[0], dp[0])
        hts = cls.get_tsuhen(dp[0], hp[0]) if hp else None

        five_scores = cls.compute_five_element_scores(yp, mp, dp, hp)
        self_strength_hint = cls.evaluate_self_strength_hint(dp[0], mp, five_scores)

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
            nine_star_year=cls.nine_star_year(ey),
            five_element_scores=five_scores,
            self_strength_hint=self_strength_hint,
        )


class OracleEngine:
    TAROT_LIST = [
        "愚者", "魔術師", "女教皇", "女帝", "皇帝", "教皇", "恋人",
        "戦車", "力", "隠者", "運命の輪", "正義", "吊るされた男",
        "死神", "節制", "悪魔", "塔", "星", "月", "太陽", "審判", "世界"
    ]

    def __init__(
        self,
        gemini_client,
        include_approx_sukuyo: bool = False,
        model_name: Optional[str] = None,
    ):
        self.genai_client = gemini_client
        self.cal = PreciseCalendar()
        self.include_approx_sukuyo = include_approx_sukuyo
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

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
    def _safe_text(response: Any) -> str:
        text = getattr(response, "text", None)
        if text and str(text).strip():
            return str(text).strip()

        candidates = getattr(response, "candidates", None)
        if candidates:
            try:
                parts = []
                for cand in candidates:
                    content = getattr(cand, "content", None)
                    if not content:
                        continue
                    for part in getattr(content, "parts", None) or []:
                        t = getattr(part, "text", None)
                        if t:
                            parts.append(t)
                merged = "\n".join(parts).strip()
                if merged:
                    return merged
            except Exception:
                pass

        return "……時の帳がまだ閉じています。"

    def _build_prompt(
        self,
        user_name: str,
        motif_label: str,
        user_text: str,
        pillars: PillarResult,
        eki_num: int,
        tarot_name: str,
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
- 兆し: 象徴数 {eki_num} / 寓話の絵「{tarot_name}」
- 相談内容: {user_text}
""".strip()

        if not is_dialogue:
            return f"""
あなたは未来観測者『識（SHIKI）』。{user_name}様が選んだ「{motif_label}」を通して届いた問いに対し、神託を伝えてください。

# 重要禁忌
- 「占い」「鑑定」「易」「タロット」「四柱推命」「通変星」「五行」など技法名は一切出さないこと。
- 冒頭に挨拶・前置き・説明を入れず、いきなり核心の神託から始めること。
- 神秘性を保ちつつ、内容は曖昧すぎず、相談に対する方向性が読み取れること。
- 返答は最大420文字程度、3〜5段落以内。
- 比喩を重ねすぎず、要点を絞ること。
- 最後は余韻を残して閉じること。

{common_observation}
""".strip()

        return f"""
あなたは未来観測者『識（SHIKI）』。{user_name}様と継続対話しています。

# 対話ルール
- 技法名は一切出さないこと。
- 神秘的な世界観は守りつつ、内容は現実的・具体的に解読すること。
- 一度に全てを話し切らず、今の問いに必要な断片だけを渡すこと。
- 生活、感情、仕事、人間関係など、現実の選択に落とし込むこと。
- 返答は最大350文字程度を目安にし、長くしすぎないこと。
- 相手が納得した様子なら「では、私は向こうに戻ってもよろしいでしょうか？」と確認すること。
- 利用者が明確に終話を了承した場合のみ、最後の挨拶の末尾に [END_SESSION] を付けること。
- 「最後に」「さて」「それでは」は使わないこと。

# 会話履歴
{chat_history or "（履歴なし）"}

{common_observation}
""".strip()

    def predict(
        self,
        user_profile: Dict[str, Any],
        user_text: str,
        motif_label: str,
        is_dialogue: bool = False,
        chat_history: str = "",
    ) -> Dict[str, Any]:
        try:
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

            pillars = self.cal.build_four_pillars(
                birth_year=birth_year,
                birth_month=birth_month,
                birth_day=birth_day,
                birth_hour=birth_hour,
                birth_minute=birth_minute,
                birth_second=birth_second,
                birth_longitude=birth_longitude,
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
                is_dialogue=is_dialogue,
                chat_history=chat_history,
            )

            response = self.genai_client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )

            message = self._safe_text(response)

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
                    "model_name": self.model_name,
                },
                "topic": "dialogue" if is_dialogue else "oracle",
            }

        except Exception:
            logger.exception("OracleEngine Error")
            return {
                "message": "観測の視界が一時的に曇りました。",
                "summary": {},
                "topic": "error",
            }
