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

class PreciseCalendar:
    """精密な東洋占術計算クラス（四柱推命・九星・暦）"""
    JUKKAN = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
    JUNISHI = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]
    KIGAKU_LIST = ["一白水星", "二黒土星", "三碧木星", "四緑木星", "五黄土星", "六白金星", "七赤金星", "八白土星", "九紫火星"]

    @staticmethod
    def julian_day(year, month, day, hour=12, minute=0, second=0, longitude=135.0):
        y, m = (year - 1, month + 12) if month <= 2 else (year, month)
        a = y // 100
        b = 2 - a + (a // 4)
        local_hour = hour + minute / 60.0 + second / 3600.0
        corrected_hour = local_hour + (longitude - 135.0) * (4.0 / 60.0)
        return math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + day + (corrected_hour / 24.0) + b - 1524.5

    @staticmethod
    def solar_longitude(jd):
        T = (jd - 2451545.0) / 36525.0
        L = (280.46646 + 36000.76983 * T) % 360.0
        M = (357.52911 + 35999.05029 * T) % 360.0
        C = (1.914602 - 0.004817 * T) * math.sin(math.radians(M)) + 0.019993 * math.sin(math.radians(2 * M))
        return (L + C) % 360.0

    @classmethod
    def build_four_pillars(cls, birth_year, birth_month, birth_day, birth_hour=None):
        jd = cls.julian_day(birth_year, birth_month, birth_day, birth_hour or 12)
        sl = cls.solar_longitude(jd)
        ey = birth_year if sl >= 315.0 else birth_year - 1
        yp = cls.JUKKAN[(ey-4)%10] + cls.JUNISHI[(ey-4)%12]
        mi = int(((sl-315.0)%360.0)//30.0)
        mp = cls.JUKKAN[({"甲":2,"己":2,"乙":4,"庚":4,"丙":6,"辛":6,"丁":8,"壬":8,"戊":0,"癸":0}[yp[0]]+mi)%10] + cls.JUNISHI[(2+mi)%12]
        di = (int(math.floor(jd+0.5)) - 2445733) % 60
        dp = cls.JUKKAN[di%10] + cls.JUNISHI[di%12]
        hp = (cls.JUKKAN[({ "甲":0,"己":0,"乙":2,"庚":2,"丙":4,"辛":4,"丁":6,"壬":6,"戊":8,"癸":8 }[dp[0]]+((birth_hour+1)//2)%12)%10] + cls.JUNISHI[((birth_hour+1)//2)%12]) if birth_hour is not None else None
        return PillarResult(year_pillar=yp, month_pillar=mp, day_pillar=dp, hour_pillar=hp, effective_year=ey, solar_longitude=sl, nine_star_year=cls.KIGAKU_LIST[(11-(ey%9)-1)%9])

class OracleEngine:
    TAROT_LIST = ["愚者", "魔術師", "女教皇", "女帝", "皇帝", "教皇", "恋人", "戦車", "力", "隠者", "運命の輪", "正義", "吊るされた男", "死神", "節制", "悪魔", "塔", "星", "月", "太陽", "審判", "世界"]

    def __init__(self, gemini_client, model_name=None):
        self.genai_client = gemini_client
        self.cal = PreciseCalendar()
        self.model_name = model_name or os.getenv("CHAT_MODEL", "gemini-1.5-flash")

    def predict(self, user_profile, user_text, motif_label, is_dialogue=False, chat_history=""):
        try:
            user_name = user_profile.get("name", "あなた")
            p = self.cal.build_four_pillars(user_profile["birth_year"], user_profile["birth_month"], user_profile["birth_day"], user_profile.get("birth_hour"))
            seed = int(hashlib.sha256(f"{user_profile['birth_year']}{motif_label}{user_text}".encode()).hexdigest(), 16)
            eki_num, tarot_name = (seed % 64) + 1, self.TAROT_LIST[seed % len(self.TAROT_LIST)]

            if not is_dialogue:
                # 【初回：純粋神託モード】
                prompt = f"""
あなたは未来観測者『識（SHIKI）』。{user_name}様が選んだ「{motif_label}」を通して届いた問いに対し、神託を伝えてください。

# 重要禁忌
- 「占い」「鑑定」「易」「タロット」「四柱推命」等の言葉は一切出さないこと。
- 冒頭に「さて」「それでは」「、最後に」等の接続詞、挨拶、前置きを絶対に入れない。いきなり神託の言葉から始めること。
- ここでは現代語での現実的な解説は行わず、比喩と象徴を用いた神秘的な余韻で終わらせること。

# 観測断片
- 宿命: {p.year_pillar} / {p.month_pillar} / {p.day_pillar}
- 兆し: 象徴数 {eki_num} / 寓話の絵「{tarot_name}」
- 相談内容（またはその背景）: {user_text}
""".strip()
            else:
                # 【継続：現実的解読・カウンセリングモード】
                prompt = f"""
あなたは未来観測者『識（SHIKI）』。{user_name}様と対話しています。

# 対話の心得
1. 答えを一度に全て話さず、相手の言葉に応じて「神託の断片」を一つずつ、現実的かつ具体的に解読してください（生活習慣、心構え、具体的アクションなど）。
2. 占術名は出さず、「今の{user_name}様にはこのような風が吹いています」と世界観を守りながら導いてください。
3. 利用者がなんとなく理解した様子や、終わりに向けた発言をした場合、すぐに会話を終わらせず、「では、私は向こうに戻ってもよろしいでしょうか？」と最終確認を行ってください。
4. その最終確認に対し、利用者が「はい」「大丈夫」など明確に同意した場合は、最後のお別れの挨拶をし、必ず文章の末尾に [END_SESSION] という文字列を出力してください。
   （もし利用者が「もう少し聞きたい」等と言った場合は、そのまま解説やカウンセリングを続けてください。[END_SESSION]は出力しないでください）
5. 冒頭や文中に「最後に」「、最後に」という言葉は絶対に使わないでください。

# 会話の文脈
{chat_history}

{user_name}様の言葉: {user_text}
""".strip()

            response = self.genai_client.models.generate_content(model=self.model_name, contents=prompt)
            return {"message": getattr(response, "text", "……時が止まったようです。")}
        except Exception as e:
            logger.exception("OracleEngine Error")
            return {"message": "観測の視界が一時的に曇りました。"}
