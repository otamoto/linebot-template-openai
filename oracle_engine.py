from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class EngineState:
    method_weights: Dict[str, float] = field(default_factory=lambda: {
        "astrology": 1.0,
        "bazi": 1.0,
        "kyusei": 1.0,
    })
    layer_weights: Dict[str, float] = field(default_factory=lambda: {
        "identity": 1.0,
        "temporal": 1.0,
        "symbolic": 1.15,
        "context": 1.25,
        "memory": 1.0,
        "branch": 1.1,
    })
    samples: int = 0
    backtest_score: float = 0.50


class TopicClassifier:
    def __init__(self):
        self.topic_keywords = {
            "love": [
                "恋", "恋愛", "好き", "彼", "彼女", "復縁", "片思い", "結婚",
                "連絡", "告白", "会いたい", "別れ", "婚活", "相手", "既読無視"
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

        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "relationship"


class RevelationEngine:
    def score_to_phase(self, action_score: float, risk_score: float) -> str:
        if risk_score > 0.76:
            return "無理に進むより、いったん流れを静めた方がいい時です"
        if action_score > 0.70 and risk_score < 0.45:
            return "流れが開きやすく、小さな動きが形になりやすい時です"
        if action_score < 0.48 and risk_score < 0.60:
            return "今は動くより整えることで、次の輪郭が見えてくる時です"
        return "止まって見えても、内側では流れが少しずつ組み替わっています"

    def risk_to_warning(self, topic: str, risk_score: float, noise_score: float) -> str:
        if topic == "love":
            if noise_score > 0.72:
                return "今は相手そのものより、不安に引っぱられて判断しやすい流れがあります。"
            if risk_score > 0.72:
                return "答えを急ぎすぎると、縁の輪郭が見えにくくなりやすい時です。"
            return "縁そのものは閉じていませんが、押し方を誤ると細くなりやすい気配があります。"

        if topic == "work":
            if risk_score > 0.72:
                return "今は努力不足というより、消耗の蓄積が判断を鈍らせやすい時です。"
            if noise_score > 0.72:
                return "問題そのものより、疲れが状況を重く見せている部分があります。"
            return "状況は固まって見えても、見方を変えると出口はまだ残っています。"

        if risk_score > 0.72:
            return "相手や環境よりも、心の疲れが関係の見え方をゆがめやすい時です。"
        return "今の関係は、切れるというより距離の取り方で形が変わる可能性があります。"

    def action_guidance(self, action_label: str) -> str:
        mapping = {
            "wait": "今は結論を急ぐより、ひとつだけ整えてから次を見る方が流れに合っています。",
            "move": "完全に止まるより、小さく動いた方が流れに噛み合いやすいです。",
            "rest": "まず心と体の消耗を落とすことが先です。休むこと自体が流れを戻します。",
            "soft_contact": "強く押すより、やわらかく気配を見せるくらいがちょうどいい時です。",
            "distance": "無理につなぐより、少し距離を置いた方が本当の輪郭が見えやすくなります。"
        }
        return mapping.get(action_label, "今は急いで答えを取りにいくより、自分の内側を整える方が次につながります。")

    def build_message(self, topic: str, scores: Dict[str, float], is_paid: bool = False) -> str:
        action_score = scores["action_score"]
        risk_score = scores["risk_score"]
        noise_score = scores["noise_score"]
        action_label = scores["best_action"]

        opening_map = {
            "love": "今の恋の流れには、少し霧があります。",
            "work": "今の仕事の流れは、静かに形を変え始めています。",
            "relationship": "今の縁は、切れるというより揺れている状態です。",
        }

        opening = opening_map.get(topic, "今の流れは、静かに揺れています。")
        phase = self.score_to_phase(action_score, risk_score)
        warning = self.risk_to_warning(topic, risk_score, noise_score)
        guidance = self.action_guidance(action_label)

        tail = (
            "見えるものだけで決めるより、まだ見えていない流れごと受け取った方が道はきれいにつながります。"
            if is_paid
            else
            "無理に未来を掴みにいくより、流れの癖を見抜いた者から道は開きます。"
        )

        return (
            f"{opening}\n\n"
            f"今のあなたは、{phase}\n"
            f"{warning}\n"
            f"{guidance}\n\n"
            f"{tail}"
        )


class OracleEngine:
    def __init__(self, state: EngineState):
        self.state = state
        self.topic_classifier = TopicClassifier()
        self.revelation_engine = RevelationEngine()

    def identity_layer(self, user_profile: Dict[str, Any]) -> Dict[str, float]:
        return {
            "resilience": float(user_profile.get("resilience", 0.55)),
            "sensitivity": float(user_profile.get("sensitivity", 0.70)),
            "patience": float(user_profile.get("patience", 0.45)),
        }

    def temporal_layer(self, horizon: str) -> Dict[str, float]:
        horizon_map = {
            "today": 0.55,
            "3days": 0.58,
            "week": 0.64,
            "month": 0.68,
        }
        return {"time_openness": horizon_map.get(horizon, 0.55)}

    def symbolic_layer(self, user_profile: Dict[str, Any]) -> Dict[str, float]:
        birth_year = int(user_profile.get("birth_year", 1990))
        birth_month = int(user_profile.get("birth_month", 6))
        birth_day = int(user_profile.get("birth_day", 15))

        astrology_flux = 0.40 + ((birth_month % 6) * 0.07)
        bazi_stability = 0.35 + (((birth_day + 2) % 7) * 0.07)
        kyusei_motion = 0.38 + (((birth_year + birth_month + birth_day) % 9) * 0.045)

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

        birth_signature = ((birth_year % 10) * 0.03) + ((birth_month % 4) * 0.04) + ((birth_day % 5) * 0.03)
        symbolic_balance = min(symbolic_balance + birth_signature, 1.0)

        return {
            "astrology_flux": astrology_flux,
            "bazi_stability": bazi_stability,
            "kyusei_motion": kyusei_motion,
            "symbolic_balance": symbolic_balance,
        }

    def context_layer(self, context_feats: Dict[str, Any]) -> Dict[str, float]:
        stress = float(context_feats.get("stress", 0.5))
        sleep_deficit = float(context_feats.get("sleep_deficit", 0.5))
        loneliness = float(context_feats.get("loneliness", 0.5))
        urgency = float(context_feats.get("urgency", 0.5))

        risk_score = (
            stress * 0.35
            + sleep_deficit * 0.25
            + loneliness * 0.20
            + urgency * 0.20
        ) * self.state.layer_weights["context"]

        risk_score = max(0.0, min(risk_score, 1.0))
        return {
            "risk_score": risk_score,
            "stress": stress,
            "sleep_deficit": sleep_deficit,
            "loneliness": loneliness,
            "urgency": urgency,
        }

    def memory_layer(self, memory: Dict[str, Any]) -> Dict[str, float]:
        repeat_count = int(memory.get("repeat_count", 1))
        volatility = float(memory.get("volatility", 0.55))
        repeat_pressure = min(repeat_count / 10.0, 1.0)
        return {
            "repeat_pressure": repeat_pressure,
            "volatility": volatility,
        }

    def branch_layer(
        self,
        topic: str,
        identity: Dict[str, float],
        temporal: Dict[str, float],
        symbolic: Dict[str, float],
        context: Dict[str, float],
        memory: Dict[str, float],
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
            resilience * 0.22
            + patience * 0.18
            + time_openness * 0.16
            + symbolic_balance * 0.22
            + (1.0 - risk_score) * 0.22
        )

        noise_score = (
            sensitivity * 0.30
            + urgency * 0.25
            + volatility * 0.25
            + repeat_pressure * 0.20
        )

        action_score = max(0.0, min(action_score, 1.0))
        noise_score = max(0.0, min(noise_score, 1.0))

        if risk_score > 0.76:
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
            "best_action": best_action,
        }

    def build_summary(self, topic: str, scores: Dict[str, float]) -> Dict[str, str]:
        action_score = scores["action_score"]
        risk_score = scores["risk_score"]

        phase = self.revelation_engine.score_to_phase(action_score, risk_score)
        risk_hint = self.revelation_engine.risk_to_warning(topic, risk_score, scores["noise_score"])
        action_hint = self.revelation_engine.action_guidance(scores["best_action"])

        if topic == "love":
            core_meaning = "不安が強くなるほど、相手そのものより自分の揺れに引っぱられやすい流れです。"
            metaphor = "霧"
            oracle_phase = "縁の輪郭が曇りやすい時期"
        elif topic == "work":
            core_meaning = "努力不足というより、疲れの蓄積が判断を重くしている流れです。"
            metaphor = "重い潮流"
            oracle_phase = "整え直しの時期"
        else:
            core_meaning = "関係そのものより、距離や受け取り方で見え方が変わりやすい流れです。"
            metaphor = "揺れる水面"
            oracle_phase = "関係の再調整期"

        return {
            "phase": phase,
            "core_meaning": core_meaning,
            "action_hint": action_hint,
            "risk_hint": risk_hint,
            "oracle_metaphor": metaphor,
            "oracle_phase": oracle_phase,
        }

    def predict(
        self,
        user_profile: Dict[str, Any],
        context_feats: Dict[str, Any],
        user_text: str,
        horizon: str = "today",
        memory: Dict[str, Any] | None = None,
        is_paid: bool = False,
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
            memory=memory_scores,
        )

        message = self.revelation_engine.build_message(topic=topic, scores=scores, is_paid=is_paid)
        summary = self.build_summary(topic=topic, scores=scores)

        self.state.samples += 1

        return {
            "topic": topic,
            "scores": scores,
            "message": message,
            "summary": summary,
            "engine_version": "ORACLE-v2.0-hybrid",
        }
