import math
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

class PreciseCalendar:
    """厳密な東洋占術計算のための暦法エンジン"""
    JUKKAN = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
    JUNISHI = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]
    SUKUYO_LIST = ["角", "亢", "氐", "房", "心", "尾", "箕", "斗", "女", "虚", "危", "室", "壁", "奎", "婁", "胃", "昴", "畢", "觜", "参", "井", "鬼", "柳", "星", "張", "翼", "軫"]
    KIGAKU_LIST = ["一白水星", "二黒土星", "三碧木星", "四緑木星", "五黄土星", "六白金星", "七赤金星", "八白土星", "九紫火星"]

    @staticmethod
    def get_solar_longitude(jd: float) -> float:
        """ユリウス日から太陽黄経を算出（高精度近似式）"""
        d = jd - 2451545.0
        g = (357.529 + 0.98560028 * d) % 360
        q = (280.459 + 0.98564736 * d) % 360
        l = (q + 1.915 * math.sin(math.radians(g)) + 0.020 * math.sin(math.radians(2 * g))) % 360
        return l

    @classmethod
    def get_sexagenary_cycle(cls, jd: float) -> str:
        """日柱（日の干支）を算出"""
        # JD 2451545.0 (2000/01/01) = 甲子(50番目スタート)
        offset = int(math.floor(jd + 0.5) - 2451545 + 50) % 60
        return cls.JUKKAN[offset % 10] + cls.JUNISHI[offset % 12]

    @classmethod
    def get_month_pillar(cls, year_kan: str, solar_long: float) -> str:
        """太陽黄経(節切り)と五虎遁法による月柱の特定"""
        # 節入り(立春=315度)からの月インデックス
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
        """地方時補正(真太陽時への近似)を含むユリウス日計算"""
        # 経度による時差補正 (1度あたり4分)
        corrected_h = h + (mn / 60.0) + (lng - 135.0) * (4.0 / 60.0)
        if m <= 2:
            y -= 1
            m += 12
        a = math.floor(y / 100)
        b = 2 - a + math.floor(a / 4)
        jd = math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + d + (corrected_h / 24.0) + b - 1524.5
        return jd

    def generate_raw_observations(self, profile: dict, motif_id: str) -> dict:
        """7つの占術データを物理・統計・直感から生成"""
        y, m, d = profile['birth_year'], profile['birth_month'], profile['birth_day']
        h, mn = profile.get('birth_hour', 12), profile.get('birth_min', 0)
        lng = profile.get('longitude', 135.0)

        jd = self._calc_julian_day(y, m, d, h, mn, lng)
        solar_long = self.cal.get_solar_longitude(jd)
        
        # 1. 四柱推命 (精密節切り判定)
        day_pillar = self.cal.get_sexagenary_cycle(jd)
        year_idx = (y - 4) % 60
        year_pillar = self.cal.JUKKAN[year_idx % 10] + self.cal.JUNISHI[year_idx % 12]
        month_pillar = self.cal.get_month_pillar(year_pillar, solar_long)

        # 2. 九星気学 (本命星)
        kigaku_idx = (12 - (y % 9)) % 9 # 簡易計算式
        kigaku = self.cal.KIGAKU_LIST[kigaku_idx]

        # 3. 宿曜 (月の離角近似)
        # 月の平均公転周期を用いた宿曜特定
        moon_jd_base = 2451550.1 # 2000年新月付近
        moon_period = 27.321661
        sukuyo_idx = int(((jd - moon_jd_base) % moon_period) / moon_period * 27)
        sukuyo = self.cal.SUKUYO_LIST[sukuyo_idx % 27]

        # 4. ユーザーの直感をシードにした「易」と「タロット」
        seed_str = f"{jd}{motif_id}{y}{m}{d}"
        seed_hash = hashlib.sha256(seed_str.encode()).hexdigest()
        seed_int = int(seed_hash, 16)

        eki_num = (seed_int % 64) + 1
        tarot_list = ["魔術師", "女教皇", "女帝", "皇帝", "教皇", "恋人", "戦車", "正義", "隠者", "運命の輪", "力", "吊るされた男", "死神", "節制", "悪魔", "塔", "星", "月", "太陽", "審判", "世界", "愚者"]
        tarot_card = tarot_list[seed_int % 22]

        return {
            "observation_point": {
                "four_pillars": {"year": year_pillar, "month": month_pillar, "day": day_pillar},
                "sukuyo": sukuyo,
                "kigaku": kigaku,
                "solar_longitude": f"{solar_long:.2f}°",
                "iching": f"第{eki_num}卦",
                "tarot": tarot_card,
                "selected_motif": motif_id
            }
        }

    def predict(self, user_profile: dict, context_feats: dict, user_text: str, motif_id: str) -> dict:
        """生データから『識』の神託を生成"""
        raw = self.generate_raw_observations(user_profile, motif_id)
        obs = raw["observation_point"]

        # 識へのシステムプロンプト (人格の注入)
        # ※この部分は次のステップでさらに強化可能
        prompt = f"""
あなたは運命観測者『識（SHIKI）』。
以下の観測事実を読み解き、一つの統合された神託を授けよ。

【観測事実】
- 四柱干支: {obs['four_pillars']['year']}年 {obs['four_pillars']['month']}月 {obs['four_pillars']['day']}日生まれ
- 宿曜: {obs['sukuyo']} / 九星: {obs['kigaku']}
- 太陽黄経: {obs['solar_longitude']}
- 刻の兆し: 易{obs['iching']} / タロット「{obs['tarot']}」
- ユーザーが選んだ象徴: {obs['selected_motif']}

【ユーザーの現状】
{user_text}

【制約】
- 占術名は出すな。
- ユーザーが選んだ「{obs['selected_motif']}」が運命の鍵となったことを冒頭で触れよ。
- 威厳のある神秘的な口調を維持せよ。
"""
        try:
            # Gemini 2.0 Flash を使用して高速かつ高精度な人格生成
            response = self.genai_client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt
            )
            return {
                "message": response.text,
                "summary": {"core_meaning": "観測完了"}, 
                "topic": "general"
            }
        except Exception as e:
            return {"message": "識の視界が一時的に曇りました。時間を置いてください。", "summary": {}, "topic": "error"}
