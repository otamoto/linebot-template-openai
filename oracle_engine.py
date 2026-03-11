from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class EngineState:
    method_weights: Dict[str, float] = field(default_factory=lambda: {
        "astrology": 1.0,
        "bazi": 1.0,
        "kyusei": 1.0
    })
    layer_weights: Dict[str, float] = field(default_factory=lambda: {
        "identity": 1.0,
        "temporal": 1.0,
        "symbolic": 1.1,
        "context": 1.3,
        "memory": 1.1,
        "branch": 1.2
    })
    samples: int = 0
    backtest_score: float = 0.50


class TopicClassifier:
    def __init__(self):
        self.topic_keywords = {
            "love": [
                "恋", "恋愛", "好き", "彼", "彼女", "復縁", "片思い", "結婚",
                "連絡", "告白", "会いたい", "別れ", "婚活", "相手"
            ],
            "work": [
                "仕事", "会社", "上司", "部下", "転職", "退職", "職場",
                "残業", "給与", "給料", "収入", "評価", "同僚", "疲れた"
            ],
            "relationship": [
                "人間関係", "友達", "家族", "仲", "距離", "喧嘩", "孤独",
                "苦しい", "嫌われた", "不安", "関係", "付き合い"
            ]
        }

    def classify(self, text: str) -> str:
        scores = {"love": 0, "work": 0, "relationship": 0}

        for topic, keywords in self.topic_keywords.items():
            for kw in keywords:
                if kw in text:
                    scores[topic] += 1

        best_topic = max(scores, key=scores.get)
        if scores[best_topic] == 0:
            return "relationship"
        return best_topic


class QuestionEngine:
    def get_question_set(self, topic: str, is_paid: bool = False) -> List[str]:
        common_questions = [
            "その悩みって、急に強くなった感じですか？ それとも前からずっと続いていましたか？",
            "今の気持ちにいちばん近いのはどれですか？ 焦り / 悲しさ / イライラ / 何も感じない感じ"
        ]

        topic_questions = {
            "love": [
                "相手との距離は今どうですか？ 近い / 少し離れている / かなり離れている",
                "今のあなたの気持ちはどれに近いですか？ 追いたい / 様子を見たい / もう終わらせたい"
            ],
            "work": [
                "いちばんしんどいのはどれですか？ 仕事量 / 人間関係 / 将来の不安",
                "今の気持ちはどちらに近いですか？ 辞めたい / もう少し耐えて様子を見たい"
            ],
            "relationship": [
                "その相手は誰に近いですか？ 家族 / 友人 / 職場 / 恋人",
                "今の気持ちはどちらに近いですか？ 仲直りしたい / 少し距離を置きたい"
            ]
        }

        if is_paid:
            # 有料は共通2問 + テーマ2問 = 4問では重いので、
            # 共通2問 + テーマ1〜2問の計3問にする
            return [common_questions[0], common_questions[1], topic_questions.get(topic, ["今の状況をもう少し詳しく教えてください。"])[0]]
        else:
            # 無料は2問
            return [common_questions[0], topic_questions.get(topic, ["今の状況をもう少し詳しく教えてください。"])[0]]


class RevelationEngine:
    def score_to_phase(self, action_score: float, risk_score: float) -> str:
        if risk_score > 0.72:
            return "少し無理をすると、流れが崩れやすい時です"
        elif action_score > 0.68 and risk_score < 0.45:
            return "流れが開きやすく、動いたことが形になりやすい時です"
        elif action_score < 0.45 and risk_score < 0.55:
            return "今は大きく動くより、整えることで次が見えてくる時です"
        else:
            return "止まっているように見えても、内側では流れが変わり始めています"

    def risk_to_warning(self, topic: str, risk_score: float, noise_score: float) -> str:
        if topic == "love":
            if noise_score > 0.7:
                return "今は相手そのものより、不安に引っぱられて判断しやすい流れがあります。"
            if risk_score > 0.7:
                return "答えを急ぎすぎると、縁の輪郭が見えにくくなりやすい時です。"
            return "縁そのものは閉じていませんが、押し方を間違えると細くなりやすい気配があります。"

        if topic == "work":
            if risk_score > 0.7:
                return "今は頑張り不足というより、疲れの積み重なりが判断を鈍らせやすい時です。"
            if noise_score > 0.7:
                return "問題そのものより、今の消耗が大きく見せている部分があります。"
            return "状況は固まっているように見えても、見方を変えると出口が見え始める流れです。"

        if risk_score > 0.7:
            return "相手や環境よりも、心の疲れが関係の見え方をゆがめやすい時です。"
        return "今の関係は、切れるというより距離の取り方で形が変わる可能性があります。"

    def action_guidance(self, action_label: str, topic: str) -> str:
        guidance_map = {
            "wait": "今日は結論を急ぐより、ひとつだけ整えてから次を見る方が流れに合っています。",
            "move": "今は完全に止まるより、小さく動いた方が流れに噛み合いやすいです。",
            "rest": "まず心と体の消耗を落とすことが先です。休むこと自体が流れを戻します。",
            "soft_contact": "強く押すより、やわらかく気配を見せるくらいがちょうどいい時です。",
            "distance": "無理に関係をつなぐより、少し距離を置くことで本当の輪郭が見えやすくなります。"
        }
        return guidance_map.get(action_label, "今は急いで答えを取りにいくより、自分の内側を整える方が次につながります。")

    def build_message(self, topic: str, scores: Dict[str, float], date_str: str, is_paid: bool = False) -> str:
        action_score = scores["action_score"]
        risk_score = scores["risk_score"]
        noise_score = scores["noise_score"]
        action_label = scores["best_action"]

        phase = self.score_to_phase(action_score, risk_score)
        warning = self.risk_to_warning(topic, risk_score, noise_score)
        guidance = self.action_guidance(action_label, topic)

        opening_map = {
            "love": [
                "今の恋の流れには、少し霧があります。",
                "今は恋の行方よりも、心の揺れが強く出やすい時です。",
                "この恋は止まっているようで、内側ではまだ動いています。"
            ],
            "work": [
                "今の仕事の流れは、静かに形を変え始めています。",
                "今は結果そのものより、疲れの影が判断に混じりやすい時です。",
                "仕事の流れは止まって見えても、水面下では少しずつ変わっています。"
            ],
            "relationship": [
                "今の縁は、切れるというより揺れている状態です。",
                "人との流れは、押すより整えることで輪郭が見えやすくなる時です。",
                "今の関係には、言葉より先に気配のずれが出ています。"
            ]
        }

        openings = opening_map.get(topic, ["今の流れは、静かに揺れています。"])
        opening = openings[0]

        if is_paid:
            tail = "見えるものだけで決めるより、まだ見えていない流れごと受け取った方が道はきれいにつながります。"
        else:
            tail = "無理に未来を掴みにいくより、流れの癖を見抜いた者から道は開きます。"

        return (
            f"{opening}\n\n"
            f"今のあなたは、{phase}。\n"
            f"{warning}\n"
            f"{guidance}\n\n"
            f"{tail}"
        )


class OracleEngine:
    def __init__(self, state: EngineState):
        self.state = state
        self.topic_classifier = TopicClassifier()
        self.question_engine = QuestionEngine()
        self.revelation_engine = RevelationEngine()

    def identity_layer(self, user_profile: Dict[str, Any]) -> Dict[str, float]:
        resilience = float(user_profile.get("resilience", 0.5))
        sensitivity = float(user_profile.get("sensitivity", 0.5))
        patience = float(user_profile.get("patience", 0.5))

        return {
            "resilience": resilience,
            "sensitivity": sensitivity,
            "patience": patience
        }

    def temporal_layer(self, horizon: str) -> Dict[str, float]:
        horizon_map = {
            "today": 0.55,
            "3days": 0.58,
            "week": 0.62,
            "month": 0.66
        }
        return {"time_openness": horizon_map.get(horizon, 0.55)}

    def symbolic_layer(self, user_profile: Dict[str, Any]) -> Dict[str, float]:
        birth_month = int(user_profile.get("birth_month", 6))

        astrology_flux = 0.45 + ((birth_month % 6) * 0.06)
        bazi_stability = 0.40 + (((birth_month + 2) % 5) * 0.08)
        kyusei_motion = 0.42 + (((birth_month + 4) % 7) * 0.05)

        total_weight = (
            self.state.method_weights["astrology"]
            + self.state.method_weights["bazi"]
            + self.state.method_weights["kyusei"]
        )

        symbolic_balance = (
            astrology_flux * self.state.method_weights["astrology"]
            + bazi_stability * self.state.method_weights["bazi"]
            + kyusei_motion * self.state.method_weights["kyusei"]
        ) / total_weight

        return {
            "astrology_flux": astrology_flux,
            "bazi_stability": bazi_stability,
            "kyusei_motion": kyusei_motion,
            "symbolic_balance": symbolic_balance
        }

    def context_layer(self, context_feats: Dict[str, Any]) -> Dict[str, float]:
        stress = float(context_feats.get("stress", 0.5))
        sleep_deficit = float(context_feats.get("sleep_deficit", 0.5))
        loneliness = float(context_feats.get("loneliness", 0.5))
        urgency = float(context_feats.get("urgency", 0.5))

        risk_score = (
            stress * 0.35 +
            sleep_deficit * 0.25 +
            loneliness * 0.20 +
            urgency * 0.20
        ) * self.state.layer_weights["context"]

        risk_score = max(0.0, min(risk_score, 1.0))

        return {
            "risk_score": risk_score,
            "stress": stress,
            "sleep_deficit": sleep_deficit,
            "loneliness": loneliness,
            "urgency": urgency
        }

    def memory_layer(self, memory: Dict[str, Any]) -> Dict[str, float]:
        repeat_count = int(memory.get("repeat_count", 1))
        volatility = float(memory.get("volatility", 0.5))
        repeat_pressure = min(repeat_count / 10.0, 1.0)

        return {
            "repeat_pressure": repeat_pressure,
            "volatility": volatility
        }

    def apply_observation_answer(self, context_feats: Dict[str, Any], answer_text: str) -> Dict[str, Any]:
        updated = dict(context_feats)
        text = answer_text.strip()

        if "焦り" in text or "急い" in text or "不安" in text:
            updated["urgency"] = min(float(updated.get("urgency", 0.5)) + 0.15, 1.0)

        if "悲しい" in text or "寂しい" in text or "孤独" in text:
            updated["loneliness"] = min(float(updated.get("loneliness", 0.5)) + 0.15, 1.0)

        if "怒" in text or "イライラ" in text:
            updated["stress"] = min(float(updated.get("stress", 0.5)) + 0.15, 1.0)

        if "空虚" in text or "虚しい" in text or "何もしたくない" in text:
            updated["stress"] = min(float(updated.get("stress", 0.5)) + 0.10, 1.0)
            updated["loneliness"] = min(float(updated.get("loneliness", 0.5)) + 0.10, 1.0)

        if "待ち" in text or "様子を見たい" in text:
            updated["urgency"] = max(float(updated.get("urgency", 0.5)) - 0.10, 0.0)

        if "動きたい" in text or "連絡したい" in text or "伝えたい" in text:
            updated["urgency"] = min(float(updated.get("urgency", 0.5)) + 0.10, 1.0)

        if "眠れてない" in text or "寝れてない" in text or "休めてない" in text:
            updated["sleep_deficit"] = min(float(updated.get("sleep_deficit", 0.5)) + 0.15, 1.0)

        if "ずっと" in text or "前から" in text or "長く" in text:
            updated["stress"] = min(float(updated.get("stress", 0.5)) + 0.05, 1.0)

        return updated

    def branch_layer(
        self,
        topic: str,
        identity: Dict[str, float],
        temporal: Dict[str, float],
        symbolic: Dict[str, float],
        context: Dict[str, float],
        memory: Dict[str, float]
    ) -> Dict[str, float]:
        resilience = identity["resilience"]
        sensitivity = identity["sensitivity"]
        patience = identity["patience"]

        time_openness = temporal["time_openness"]
        symbolic_balance = symbolic["symbolic_balance"]

        risk_score = context["risk_score"]
        urgency = context["urgency"]

        repeat_pressure = memory["repeat_pressure"]
        volatility = memory["volatility"]

        action_score = (
            resilience * 0.22 +
            patience * 0.18 +
            time_openness * 0.18 +
            symbolic_balance * 0.20 +
            (1.0 - risk_score) * 0.22
        )

        noise_score = (
            sensitivity * 0.30 +
            urgency * 0.25 +
            volatility * 0.25 +
            repeat_pressure * 0.20
        )

        action_score = max(0.0, min(action_score, 1.0))
        noise_score = max(0.0, min(noise_score, 1.0))

        if risk_score > 0.75:
            best_action = "rest"
        elif topic == "love" and noise_score > 0.68:
            best_action = "wait"
        elif topic == "love" and action_score > 0.62:
            best_action = "soft_contact"
        elif topic == "work" and risk_score > 0.62:
            best_action = "wait"
        elif topic == "relationship" and noise_score > 0.65:
            best_action = "distance"
        elif action_score > 0.66:
            best_action = "move"
        else:
            best_action = "wait"

        return {
            "action_score": round(action_score, 3),
            "risk_score": round(risk_score, 3),
            "noise_score": round(noise_score, 3),
            "best_action": best_action
        }

    def predict(
        self,
        user_profile: Dict[str, Any],
        context_feats: Dict[str, Any],
        user_text: str,
        date_str: str,
        horizon: str = "today",
        memory: Dict[str, Any] = None,
        is_paid: bool = False
    ) -> Dict[str, Any]:
        if memory is None:
            memory = {}

        topic = self.topic_classifier.classify(user_text)

        identity = self.identity_layer(user_profile)
        temporal = self.temporal_layer(horizon)
        symbolic = self.symbolic_layer(user_profile)
        context = self.context_layer(context_feats)
        memory_scores = self.memory_layer(memory)

        scores = self.branch_layer(
            topic=topic,
            identity=identity,
            temporal=temporal,
            symbolic=symbolic,
            context=context,
            memory=memory_scores
        )

        revelation = self.revelation_engine.build_message(
            topic=topic,
            scores=scores,
            date_str=date_str,
            is_paid=is_paid
        )

        followup_questions = self.question_engine.get_question_set(topic, is_paid=is_paid)

        self.state.samples += 1

        return {
            "topic": topic,
            "date": date_str,
            "horizon": horizon,
            "scores": scores,
            "message": revelation,
            "followup_questions": followup_questions,
            "engine_version": "ORACLE-v2-prototype"
        }
