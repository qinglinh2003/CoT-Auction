from typing import Dict

def to_token_cost(usage: Dict[str, int], price_cfg: Dict) -> Dict[str, float]:
    pin = price_cfg["input_per_1k"]
    pout = price_cfg["output_per_1k"]
    cost = (usage["in"]/1000.0)*pin + (usage["out"]/1000.0)*pout
    return {"usd": round(cost, 6)}
