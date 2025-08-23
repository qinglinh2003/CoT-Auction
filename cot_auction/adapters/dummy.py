import random, time
from typing import Dict, Any
from .base import ModelAdapter, GenConfig

class DummyAdapter(ModelAdapter):
    def generate_chains(self, prompt: str, cfg: GenConfig) -> Dict[str, Any]:
        random.seed(hash(prompt) % (2**32))  
        chains = []
        t0 = time.time()
        for _ in range(cfg.k):
            if "Yes/No" in prompt or "Yes or No" in prompt:
                ans = "Yes" if random.random() < 0.7 else "No"
            elif "YYYY-MM-DD" in prompt:
                ans = "2025-09-22"
            elif "day of the week" in prompt:
                ans = random.choice(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
            else:
                ans = str(random.choice([59,80,150,336,3500,361,378]))
            chains.append({
                "final": f"Final: {ans}",
                "brief_rationale": "Heuristic guess (dummy).",
                "usage": {"in": 30, "out": 10},   
                "latency_s": 0.02
            })
        overall_latency = max(c["latency_s"] for c in chains) if chains else (time.time() - t0)
        return {"chains": chains, "latency_s": overall_latency}
