import math
import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class PillarResult:
    year_pillar: str
    month_pillar: str
    day_pillar: str
    hour_pillar: Optional[str]
    effective_year: int
    solar_longitude: float
    nine_star_year: str
    sukuyo: Optional[str] = None


class PreciseCalendar:
    """
    精度重視の東洋占術ベース計算クラス

    方針:
    - 年柱: 立春（太陽黄経315°）基準
    - 月柱: 節入り（太陽黄経を30°ごと）基準
    - 日柱: 基準甲子日からJDNで算出
    - 時柱: 日干 + 時支で算出
    - 九星: 年盤の本命星を立春基準で算出
    - 宿曜: 厳密計算に必要な要素を省くため strict では未使用
    """

    JUKKAN = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
    JUNISHI = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]
    KIGAKU_LIST = [
        "一白水星", "二黒土星", "三碧木星", "四緑木星", "五黄土星",
        "六白金星", "七赤金星", "八白土星", "九紫火星"
    ]
    SUKUYO_LIST = [
        "角", "亢", "氐", "房", "心", "尾", "箕", "斗", "女",
        "虚", "危", "室", "壁", "奎", "婁", "胃", "昴", "畢",
        "觜", "参", "井", "鬼", "柳", "星", "張", "翼", "軫"
    ]

    # 1984-02-02 を甲子日基準として採用
    # floor(JD + 0.5) に対応するJDN整数
    SEXAGENARY_DAY_BASE_JDN = 2445733
    SEXAGENARY_DAY_BASE_INDEX = 0  # 甲子=0

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
        """
        地方平均太陽時寄りの補正を軽く入れたJD計算。
        入力は日本の現地時刻を想定。
        """
        local_hour = hour + minute / 60.0 + second / 3600.0

        # 東経135°基準からの地方時補正（1度=4分）
        corrected_hour = local_hour + (longitude - 135.0) * (4.0 / 60.0)

        y = year
        m = month
        if m <= 2:
            y -= 1
            m += 12

        a = y // 100
        b = 2 - a + (a // 4)

        jd = (
            math.floor(365.25 * (y + 4716))
            + math.floor(30.6001 * (m + 1))
            + day
            + (corrected_hour / 24.0)
            + b
            - 1524.5
        )
        return jd

    @staticmethod
    def solar_longitude(jd: float) -> float:
        """
        太陽黄経の近似計算（Meeus系の簡略式）。
        節入り判定用途としては十分実用的。
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
        """
        立春（315°）前なら前年扱い
        """
        return birth_year if solar_long >= 315.0 else birth_year - 1

    @classmethod
    def year_pillar(cls, effective_year: int) -> str:
        stem = cls.JUKKAN[(effective_year - 4) % 10]
        branch = cls.JUNISHI[(effective_year - 4) % 12]
        return stem + branch

    @classmethod
    def month_index_from_solar_longitude(cls, solar_long: float) -> int:
        """
        寅月を0として 315°(立春) 起点に30°ごと
        寅=0, 卯=1, ... , 丑=11
        """
        return int(((solar_long - 315.0) % 360.0) // 30.0)

    @classmethod
    def month_pillar(cls, year_stem: str, solar_long: float) -> str:
        """
        月柱:
        年干ごとの寅月天干起点を使用
        甲己 -> 丙寅
        乙庚 -> 戊寅
        丙辛 -> 庚寅
        丁壬 -> 壬寅
        戊癸 -> 甲寅
        """
        month_idx = cls.month_index_from_solar_longitude(solar_long)

        start_kan_map = {
            "甲": 2, "己": 2,  # 丙
            "乙": 4, "庚": 4,  # 戊
            "丙": 6, "辛": 6,  # 庚
            "丁": 8, "壬": 8,  # 壬
            "戊": 0, "癸": 0,  # 甲
        }

        start_stem_idx = start_kan_map[year_stem]
        stem = cls.JUKKAN[(start_stem_idx + month_idx) % 10]
        branch = cls.JUNISHI[(2 + month_idx) % 12]  # 寅始まり
        return stem + branch

    @classmethod
    def sexagenary_day_index(cls, jd: float) -> int:
        """
        日柱用の干支インデックス
        """
        jdn = int(math.floor(jd + 0.5))
        return (jdn - cls.SEXAGENARY_DAY_BASE_JDN + cls.SEXAGENARY_DAY_BASE_INDEX) % 60

    @classmethod
    def day_pillar(cls, jd: float) -> str:
        idx = cls.sexagenary_day_index(jd)
        return cls.JUKKAN[idx % 10] + cls.JUNISHI[idx % 12]

    @classmethod
    def hour_branch_index(cls, hour: int) -> int:
        """
        子時 23:00-00:59
        丑時 01:00-02:59 ...
        """
        return ((hour + 1) // 2) % 12

    @classmethod
    def hour_pillar(cls, day_stem: str, hour: int) -> str:
        """
        時柱:
        甲己日 -> 甲子始まり
        乙庚日 -> 丙子始まり
        丙辛日 -> 戊子始まり
        丁壬日 -> 庚子始まり
        戊癸日 -> 壬子始まり
        """
        start_map = {
            "甲": 0, "己": 0,  # 甲
            "乙": 2, "庚": 2,  # 丙
            "丙": 4, "辛": 4,  # 戊
            "丁": 6, "壬": 6,  # 庚
            "戊": 8, "癸": 8,  # 壬
        }

        hb_idx = cls.hour_branch_index(hour)
        stem_start = start_map[day_stem]
        stem = cls.JUKKAN[(stem_start + hb_idx) % 10]
        branch = cls.JUNISHI[hb_idx]
        return stem + branch

    @classmethod
    def nine_star_year(cls, effective_year: int) -> str:
        """
        九星気学の本命星（年盤）を立春基準で算出
        """
        star_num = 11 - (effective_year % 9)
        while star_num > 9:
            star_num -= 9
        if star_num <= 0:
            star_num += 9
        return cls.KIGAKU_LIST[star_num - 1]

    @classmethod
    def sukuyo_approx(cls, jd: float) -> str:
        """
        参考用の簡易近似。厳密宿曜ではない。
        """
        idx = int((((jd - 2451550.1) % 27.32166) / 27.32166) * 27) % 27
        return cls.SUKUYO_LIST[idx]

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
        include_hour_pillar: bool = True,
        include_approx_sukuyo: bool = False,
    ) -> PillarResult:
        jd = cls.julian_day(
            year=birth_year,
            month=birth_month,
            day=birth_day,
            hour=birth_hour if birth_hour is not None else 12,
            minute=birth_minute,
            second=birth_second,
            longitude=birth_longitude,
        )

        solar_long = cls.solar_longitude(jd)
        effective_year = cls.effective_year_by_setsu(birth_year, solar_long)

        year_p = cls.year_pillar(effective_year)
        month_p = cls.month_pillar(year_p[0], solar_long)
        day_p = cls.day_pillar(jd)

        hour_p = None
        if include_hour_pillar and birth_hour is not None:
            hour_p = cls.hour_pillar(day_p[0], birth_hour)

        nine_star = cls.nine_star_year(effective_year)
        sukuyo = cls.sukuyo_approx(jd) if include_approx_sukuyo else None

        return PillarResult(
            year_pillar=year_p,
            month_pillar=month_p,
            day_pillar=day_p,
            hour_pillar=hour_p,
            effective_year=effective_year,
            solar_longitude=solar_long,
            nine_star_year=nine_star,
            sukuyo=sukuyo,
        )


class OracleEngine:
    """
    精度重視版 OracleEngine

    改善点:
    - 年柱・月柱・九星を節切り基準
    - 日柱基準を明示
    - 時柱対応
    - 宿曜は厳密でない限り省略可能
    - Geminiレスポンスの安全取得
    """

    TAROT_LIST = [
        "愚者", "魔術師", "女教皇", "女帝", "皇帝", "教皇", "恋人",
        "戦車", "力", "隠者", "運命の輪", "正義", "吊るされた男",
        "死神", "節制", "悪魔", "塔", "星", "月", "太陽", "審判", "世界"
    ]

    def __init__(self, gemini_client, include_approx_sukuyo: bool = False):
        self.genai_client = gemini_client
        self.cal = PreciseCalendar()
        self.include_approx_sukuyo = include_approx_sukuyo

    @staticmethod
    def _stable_seed(
        y: int,
        m: int,
        d: int,
        h: Optional[int],
        mn: int,
        motif_id: str,
        user_text: str,
    ) -> int:
        seed_str = f"{y:04d}-{m:02d}-{d:02d}|{h}|{mn}|{motif_id}|{user_text}"
        seed_hash = hashlib.sha256(seed_str.encode("utf-8")).hexdigest()
        return int(seed_hash, 16)

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
                    content_parts = getattr(content, "parts", None) or []
                    for p in content_parts:
                        t = getattr(p, "text", None)
                        if t:
                            parts.append(t)
                merged = "\n".join(parts).strip()
                if merged:
                    return merged
            except Exception:
                pass

        return "……識はまだ言葉を結ばない。"

    def _build_prompt(
        self,
        pillars: PillarResult,
        eki_num: int,
        tarot: str,
        motif_id: str,
        user_text: str,
        context_feats: Dict[str, Any],
    ) -> str:
        lines = [
            "あなたは未来観測者『識（SHIKI）』。",
            "以下の観測情報を一つの神託として統合し、占術名を直接は出さずに語れ。",
            f"冒頭では必ず、ユーザーが選んだ象徴「{motif_id}」が運命を動かしたことに触れよ。",
            "",
            "【観測情報】",
            f"- 年柱: {pillars.year_pillar}",
            f"- 月柱: {pillars.month_pillar}",
            f"- 日柱: {pillars.day_pillar}",
        ]

        if pillars.hour_pillar:
            lines.append(f"- 時柱: {pillars.hour_pillar}")

        lines.append(f"- 九星: {pillars.nine_star_year}")

        if pillars.sukuyo:
            lines.append(f"- 宿曜(参考近似): {pillars.sukuyo}")

        lines.extend([
            f"- 兆し: 易第{eki_num}卦 / タロット「{tarot}」",
            f"- 問い: {user_text}",
            f"- 心理状態: {context_feats}",
            "",
            "【制約】",
            "- 「占いの結果です」「AIとして」などの説明は不要。",
            "- 静かで威厳のある、古い預言者のような口調で語ること。",
            "- 断定しすぎず、しかし曖昧逃げにもならないこと。",
            "- 最後は短く余韻を残すこと。",
        ])
        return "\n".join(lines)

    def predict(
        self,
        user_profile: Dict[str, Any],
        context_feats: Dict[str, Any],
        user_text: str,
        motif_id: str,
    ) -> Dict[str, Any]:
        try:
            y = int(user_profile["birth_year"])
            m = int(user_profile["birth_month"])
            d = int(user_profile["birth_day"])
            h = user_profile.get("birth_hour")
            mn = int(user_profile.get("birth_minute", 0))
            sec = int(user_profile.get("birth_second", 0))
            lng = float(user_profile.get("birth_longitude", 135.0))

            if h is not None:
                h = int(h)

            pillars = self.cal.build_four_pillars(
                birth_year=y,
                birth_month=m,
                birth_day=d,
                birth_hour=h,
                birth_minute=mn,
                birth_second=sec,
                birth_longitude=lng,
                include_hour_pillar=(h is not None),
                include_approx_sukuyo=self.include_approx_sukuyo,
            )

            seed_int = self._stable_seed(y, m, d, h, mn, motif_id, user_text)
            eki_num = (seed_int % 64) + 1
            tarot = self.TAROT_LIST[seed_int % len(self.TAROT_LIST)]

            prompt = self._build_prompt(
                pillars=pillars,
                eki_num=eki_num,
                tarot=tarot,
                motif_id=motif_id,
                user_text=user_text,
                context_feats=context_feats,
            )

            response = self.genai_client.models.generate_content(
                model="models/gemini-1.5-flash",
                contents=prompt,
            )
            message = self._safe_text(response)

            summary = {
                "core_meaning": "観測完了",
                "effective_year": pillars.effective_year,
                "solar_longitude": round(pillars.solar_longitude, 6),
                "year_pillar": pillars.year_pillar,
                "month_pillar": pillars.month_pillar,
                "day_pillar": pillars.day_pillar,
                "hour_pillar": pillars.hour_pillar,
                "nine_star_year": pillars.nine_star_year,
                "sukuyo": pillars.sukuyo,
                "eki_num": eki_num,
                "tarot": tarot,
            }

            return {
                "message": message,
                "summary": summary,
                "topic": "general",
            }

        except Exception as e:
            logger.exception("OracleEngine Error")
            error_msg = str(e)

            if "429" in error_msg:
                return {
                    "message": "……天の理が一時的に混み合っているようです。少し、時を置いてから再び声をかけてください。――識より",
                    "summary": {},
                    "topic": "error",
                }

            return {
                "message": f"識の視界が一時的に曇りました（{error_msg[:80]}...）",
                "summary": {},
                "topic": "error",
            }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    class DummyResponse:
        def __init__(self, text: str):
            self.text = text

    class DummyModels:
        @staticmethod
        def generate_content(model: str, contents: str):
            return DummyResponse(f"[{model}] {contents[:180]}...")

    class DummyGeminiClient:
        models = DummyModels()

    engine = OracleEngine(DummyGeminiClient(), include_approx_sukuyo=False)

    result = engine.predict(
        user_profile={
            "birth_year": 1992,
            "birth_month": 2,
            "birth_day": 3,
            "birth_hour": 23,
            "birth_minute": 40,
            "birth_longitude": 139.6917,
        },
        context_feats={
            "mood": "揺れている",
            "focus": "仕事と今後の流れ",
        },
        user_text="これから数ヶ月、私はどの選択を優先すべきか。",
        motif_id="月影の鍵",
    )

    print("summary =", result["summary"])
    print("message =", result["message"])
