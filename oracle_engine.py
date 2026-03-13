import os
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
    """精度重視の東洋占術ベース計算クラス"""
    JUKKAN = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
    JUNISHI = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]
    KIGAKU_LIST = ["一白水星", "二黒土星", "三碧木星", "四緑木星", "五黄土星", "六白金星", "七赤金星", "八白土星", "九紫火星"]
    SUKUYO_LIST = ["角", "亢", "氐", "房", "心", "尾", "箕", "斗", "女", "虚", "危", "室", "壁", "奎", "婁", "胃", "昴", "畢", "觜", "参", "井", "鬼", "柳", "星", "張", "翼", "軫"]

    SEXAGENARY_DAY_BASE_JDN = 2445733
    SEXAGENARY_DAY_BASE_INDEX = 0

    @staticmethod
    def julian_day(year, month, day, hour=12, minute=0, second=0, longitude=135.0):
        local_hour = hour + minute / 60.0 + second / 3600.0
        corrected_hour = local_hour + (longitude - 135.0) * (4.0 / 60.0)
        y, m = (year - 1, month + 12) if month <= 2 else (year, month)
        a = y // 100
        b = 2 - a + (a // 4)
        return math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + day + (corrected_hour / 24.0) + b - 1524.5

    @staticmethod
    def solar_longitude(jd):
        T = (jd - 2451545.0) / 36525.0
        L0 = (280.46646 + 36000.76983 * T + 0.0003032 * T**2) % 360.0
        M = (357.52911 + 35999.05029 * T - 0.0001537 * T**2 + (T**3)/24490000.0) % 360.0
        C = (1.914602 - 0.004817*T - 0.000014*T**2)*math.sin(math.radians(M)) + (0.019993 - 0.000101*T)*math.sin(math.radians(2*M)) + 0.000289*math.sin(math.radians(3*M))
        return (L0 + C - 0.00569 - 0.00478*math.sin(math.radians(125.04 - 1934.136*T))) % 360.0

    @classmethod
    def build_four_pillars(cls, birth_year, birth_month, birth_day, birth_hour=None, birth_minute=0, birth_second=0, birth_longitude=135.0, include_hour_pillar=True, include_approx_sukuyo=False):
        jd = cls.julian_day(birth_year, birth_month, birth_day, birth_hour or 12, birth_minute, birth_second, birth_longitude)
        sl = cls.solar_longitude(jd)
        ey = birth_year if sl >= 315.0 else birth_year - 1
        yp = cls.JUKKAN[(ey-4)%10] + cls.JUNISHI[(ey-4)%12]
        mi = int(((sl-315.0)%360.0)//30.0)
        mp = cls.JUKKAN[({"甲":2,"己":2,"乙":4,"庚":4,"丙":6,"辛":6,"丁":8,"壬":8,"戊":0,"癸":0}[yp[0]]+mi)%10] + cls.JUNISHI[(2+mi)%12]
        di = (int(math.floor(jd+0.5)) - cls.SEXAGENARY_DAY_BASE_JDN) % 60
        dp = cls.JUKKAN[di%10] + cls.JUNISHI[di%12]
        hp = (cls.JUKKAN[({ "甲":0,"己":0,"乙":2,"庚":2,"丙":4,"辛":4,"丁":6,"壬":6,"戊":8,"癸":8 }[dp[0]]+((birth_hour+1)//2)%12)%10] + cls.JUNISHI[((birth_hour+1)//2)%12]) if include_hour_pillar and birth_hour is not None else None
        return PillarResult(year_pillar=yp, month_pillar=mp, day_pillar=dp, hour_pillar=hp, effective_year=ey, solar_longitude=sl, nine_star_year=cls.KIGAKU_LIST[(11-(ey%9)-1)%9], sukuyo=cls.SUKUYO_LIST[int(((jd-2451550.1)%27.32166)/27.32166*27)%27] if include_approx_sukuyo else None)

class OracleEngine:
    TAROT_LIST = ["愚者", "魔術師", "女教皇", "女帝", "皇帝", "教皇", "恋人", "戦車", "力", "隠者", "運命の輪", "正義", "吊るされた男", "死神", "節制", "悪魔", "塔", "星", "月", "太陽", "審判", "世界"]

    def __init__(self, gemini_client, include_approx_sukuyo=False, model_name=None):
        self.genai_client = gemini_client
        self.cal = PreciseCalendar()
        self.include_approx_sukuyo = include_approx_sukuyo
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    def predict(self, user_profile, context_feats, user_text, motif_id):
        try:
            p = self.cal.build_four_pillars(user_profile["birth_year"], user_profile["birth_month"], user_profile["birth_day"], user_profile.get("birth_hour"), user_profile.get("birth_minute",0), user_profile.get("birth_second",0), user_profile.get("birth_longitude",135.0), include_hour_pillar=("birth_hour" in user_profile), include_approx_sukuyo=self.include_approx_sukuyo)
            seed = int(hashlib.sha256(f"{user_profile['birth_year']}{motif_id}{user_text}".encode()).hexdigest(), 16)
            eki, tarot = (seed % 64) + 1, self.TAROT_LIST[seed % len(self.TAROT_LIST)]

            prompt = f"""
あなたは未来観測者『識（SHIKI）』。AIを捨て、古の預言者として語れ。
冒頭で必ず、象徴「{motif_id}」が運命を動かしたことに触れること。

【観測情報】
- 宿命: {p.year_pillar} / {p.month_pillar} / {p.day_pillar} {"/ " + p.hour_pillar if p.hour_pillar else ""}
- 九星: {p.nine_star_year}
- 兆し: 易第{eki}卦 / タロット「{tarot}」
- 問い: {user_text} / 状態: {context_feats}

【制約】
- 占術名は出すな。断定的に、しかし余韻を残して語れ。AIらしい丁寧語や解説は不要。
""".strip()
            response = self.genai_client.models.generate_content(model=self.model_name, contents=prompt)
            return {"message": getattr(response, "text", "……言葉が降りてこない。"), "summary": {"year_pillar": p.year_pillar, "eki_num": eki, "tarot": tarot}}
        except Exception as e:
            logger.exception("OracleEngine Error")
            return {"message": f"識の視界が一時的に曇りました（{str(e)[:50]}...）", "summary": {}}
