import random
from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class EngineState:
    method_weights: Dict[str, float] = field(default_factory=lambda: {"astrology": 1.0, "bazi": 1.0})
    layer_weights: Dict[str, float] = field(default_factory=lambda: {"A_base": 1.0, "B_context": 1.2, "C_sim": 1.1})
    samples: int = 0
    backtest_brier: float = 0.2

class OracleEngine:
    def __init__(self, state: EngineState):
        self.state = state

    def predict(self, user_profile: dict, context_feats: dict, date_str: str) -> dict:
        base_potency = random.uniform(0.4, 0.7)
        stress = context_feats.get("stress", 0.5)
        sleep = context_feats.get("sleep_deficit", 0.5)
        risk_score = (stress * 0.4 + sleep * 0.4) * self.state.layer_weights["B_context"]
        
        n_trials = 2000
        successes = sum(1 for _ in range(n_trials) if (base_potency - (risk_score * 0.5) + random.gauss(0, 0.15)) > 0.5)
        
        probability = successes / n_trials
        self.state.samples += 1
        
        return {
            "date": date_str,
            "success_probability": round(probability * 100, 1),
            "risk_level": "high" if risk_score > 0.7 else "medium" if risk_score > 0.4 else "low",
            "engine_version": "SHIKI-v1.0"
        }
