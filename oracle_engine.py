import math
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, Any

# ログ設定
logger = logging.getLogger(__name__)

class PreciseCalendar:
    """精密な東洋占術計算（節切り・干支・宿曜）を行うエンジン"""
    JUKKAN = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
    JUNISHI = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]
    SUKUYO_LIST = ["角", "亢", "氐", "房", "心", "尾", "箕", "斗", "女", "虚", "危", "室", "壁", "奎", "婁", "胃", "昴", "畢", "觜", "参", "井", "鬼", "柳", "星", "張", "翼", "軫"]
    KIGAKU_LIST = ["一白水星", "二黒土星", "三碧木星", "四緑木星", "五黄土星", "六白金星", "七赤金星", "八白土星", "九紫火星"]

    @staticmethod
    def get_solar_longitude(jd: float) -> float:
        """ユリウス日から太陽黄経を算出"""
        d = jd - 2451545.0
        g = (357.529 + 0.98560028 * d) % 360
        q = (280.459 + 0.98564736 * d) % 360
        l = (q + 1.915 * math.sin(math.radians(g)) + 0.020 * math.sin(math.radians(2 * g))) % 360
        return l

    @classmethod
    def get_sexagenary_cycle(cls, jd: float) -> str:
        """日柱（日の干支）を算出"""
        offset = int(math.floor(jd + 0.5) - 2451545 + 50) % 60
        return cls.JUKKAN[offset % 10] + cls.JUNISHI[offset % 12]

    @classmethod
    def get_month_pillar(cls, year_kan: str, solar_long: float) -> str:
        """月柱の特定"""
        month_idx = int((solar_long - 315) % 360 / 30)
        start_kan_map = {"甲": 2, "己": 2, "乙": 4, "庚": 4, "丙": 6, "辛": 6, "丁": 8, "壬": 8, "戊": 0, "癸": 0}
        start_kan_idx = start_kan_map.get(year_kan[0], 0)
        month_kan = cls.JUKKAN[(start_kan_idx + month_idx) % 10]
        month_shi = cls.JUNISHI[(month_idx + 2) % 12]
        return month_kan + month_shi

class OracleEngine:
    def __init__(self, gemini_client):
        self.genai_client = gemini_client
        self.cal = PreciseCalendar()

    def _calc_julian_day(self, y: int, m: int, d: int, h: int, mn: int, lng: float = 135.0) -> float:
        """真太陽時補正を含むユリウス日計算"""
        corrected_h = h + (mn / 60.0) + (lng - 135.0) * (4.0 / 60.0)
        if m <= 2:
            y -= 1
            m += 12
        a = math.floor(y / 100)
        b = 2 - a + math.floor(a / 4)
        jd = math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + d + (corrected_h / 24.0) + b - 1524.5
        return jd

    def predict(self, user_profile: dict, context_feats: dict, user_text: str, motif_id: str) -> dict:
        """精密計算と『識』の神託生成"""
        try:
            y, m, d = user_profile['birth_year'], user_profile['birth_month'], user_profile['birth_day']
            h = user_profile.get('birth_hour', 12)
            
            jd = self._calc_julian_day(y, m, d, h, 0)
            solar_long = self.cal.get_solar_longitude(jd)
            
            # 1. 四柱推命
            day_pillar = self.cal.get_sexagenary_cycle(jd)
            year_pillar = self.cal.JUKKAN[(y - 4) % 10] + self.cal.JUNISHI[(y - 4) % 12]
            month_pillar = self.cal.get_month_pillar(year_pillar, solar_long)

            # 2. 宿曜・九星
            # 月の平均公転周期を用いた簡易宿曜特定
            sukuyo_idx = int(((jd - 2451550.1) % 27.32) / 27.32 * 27) % 27
            sukuyo = self.cal.SUKUYO_LIST[sukuyo_idx]
            kigaku = self.cal.KIGAKU_LIST[(12 - (y % 9)) % 9]

            # 3. 直感シード（易・タロット）
            seed_str = f"{jd}{motif_id}{y}{m}{d}"
            seed_hash = hashlib.sha256(seed_str.encode()).hexdigest()
            seed_int = int(seed_hash, 16)

            eki_num = (seed_int % 64) + 1
            tarot_list = ["魔術師", "女教皇", "女帝", "皇帝", "教皇", "恋人", "戦車", "正義", "隠者", "運命の輪", "力", "吊るされた男", "死神", "節制", "悪魔", "塔", "星", "月", "太陽", "審判", "世界", "愚者"]
            tarot = tarot_list[seed_int % 22]

            # 識の人格と観測データをプロンプト化
            prompt = f"""
あなたは未来観測者『識（SHIKI）』。以下の観測データを、占術名を出さずに一つの神託として統合せよ。
冒頭で必ずユーザーが選んだ象徴「{motif_id}」が運命を動かしたことに触れよ。

【観測データ】
- 干支: {year_pillar}年 {month_pillar}月 {day_pillar}日
- 宿曜/九星: {sukuyo} / {kigaku}
- 兆し: 易第{eki_num}卦 / タロット「{tarot}」
- ユーザーの問い: {user_text}
- 心理状態: {context_feats}

【制約】
- 「占いの結果です」などのAIらしい前置きは一切不要。
- 静かで威厳のある、古の預言者のような口調で語れ。
"""

            # モデル名を models/gemini-1.5-flash に固定して404を回避
            response = self.genai_client.models.generate_content(
                model="models/gemini-1.5-flash", 
                contents=prompt
            )
            
            return {
                "message": response.text,
                "summary": {"core_meaning": "観測完了"}, 
                "topic": "general"
            }
            
        except Exception as e:
            logger.error(f"OracleEngine Error: {e}")
            error_msg = str(e)
            if "429" in error_msg:
                return {"message": "……天の理が一時的に混み合っているようです。少し、時を置いてから再び声をかけてください。――識より", "summary": {}, "topic": "error"}
            return {"message": f"識の視界が一時的に曇りました（{error_msg[:30]}...）", "summary": {}, "topic": "error"}
