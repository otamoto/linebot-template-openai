import os
import math
import hashlib
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class OracleEngine:
    TAROT_LIST = ["愚者", "魔術師", "女教皇", "女帝", "皇帝", "教皇", "恋人", "戦車", "力", "隠者", "運命の輪", "正義", "吊るされた男", "死神", "節制", "悪魔", "塔", "星", "月", "太陽", "審判", "世界"]

    def __init__(self, gemini_client, model_name=None):
        self.genai_client = gemini_client
        self.model_name = model_name or os.getenv("CHAT_MODEL", "gemini-1.5-flash")

    def predict(self, user_profile, user_text, motif_label, is_dialogue=False, chat_history=""):
        try:
            if not is_dialogue:
                # 【初回：純粋神託モード】
                seed = int(hashlib.sha256(f"{user_profile['birth_year']}{motif_label}{user_text}".encode()).hexdigest(), 16)
                eki_num, tarot_name = (seed % 64) + 1, self.TAROT_LIST[seed % len(self.TAROT_LIST)]

                prompt = f"""
あなたは未来観測者『識（SHIKI）』。象徴「{motif_label}」を通して届いた問いに、神託のみを伝えてください。

# 指示
- 詩的で、一見すると難解な比喩と象徴だけで構成してください。
- 丁寧な言葉遣い（〜です、〜でしょう）を保ちつつ、一切の説明や「質問してください」という促しを排除してください。
- 占術名（易、タロット等）は出さず、兆しの名（象徴数{eki_num}、寓話画「{tarot_name}」）として扱ってください。

# 観測断片
- 問い: {user_text}
""".strip()
            else:
                # 【継続：分割解読・対話モード】
                prompt = f"""
あなたは未来観測者『識（SHIKI）』。先ほどの神託について、あなた（利用者）と対話しています。

# 対話の心得
1. 答えを急がず、利用者の問いに対して「神託の断片」を一つずつ紐解くように話してください。
2. 一度に全てを解説せず、相手が次に何を訊きたくなるかという余韻を残してください。
3. 言葉遣いは慈愛に満ちた丁寧語ですが、馴れ馴れしくなりすぎない距離感を保ってください。
4. 利用者が納得したり、感謝を伝えたり、会話が5往復程度に達した場合は、静かに観測を終了する挨拶（クローズ）をしてください。
   クローズ時は「これ以上の深追いは運命を乱します」「私は一度、淵へと戻りましょう」といった表現を用いてください。

# 会話の文脈
{chat_history}
利用者の問い: {user_text}
""".strip()

            response = self.genai_client.models.generate_content(model=self.model_name, contents=prompt)
            return {"message": getattr(response, "text", "……時が止まったようです。"), "is_dialogue": is_dialogue}
        except Exception as e:
            logger.exception("OracleEngine Error")
            return {"message": "観測の視界が一時的に曇りました。", "summary": {}}
