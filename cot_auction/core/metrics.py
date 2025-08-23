import re
from collections import defaultdict, Counter
from typing import List, Dict, Any

def judge_one(item: Dict[str, Any], pred: str) -> int:
    gold = item["answer"]
    mode = item.get("judge", "exact")
    if mode == "exact":
        return int(str(gold).strip() == str(pred).strip())
    elif mode == "regex":
        pat = item.get("pattern", str(gold))
        return int(re.fullmatch(pat, str(pred)) is not None)
    else:
        return int(str(gold).strip() == str(pred).strip())

def aggregate_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_k = defaultdict(list)
    by_cat = defaultdict(list)
    for r in records:
        by_k[r["k"]].append(r)
        by_cat[r["category"]].append(r)

    def summarize(lst):
        n = len(lst) or 1
        acc = sum(r["correct"] for r in lst) / n
        avg_cost = sum(r["cost_usd"] for r in lst) / n
        avg_lat = sum(r["latency_s"] for r in lst) / n
        avg_agree = sum(r["agreement"] for r in lst) / n
        return {"n": n, "acc": acc, "avg_cost_usd": avg_cost, "avg_latency_s": avg_lat, "avg_agreement": avg_agree}

    overall = summarize(records)
    return {
        "overall": overall,
        "by_k": {int(k): summarize(v) for k, v in by_k.items()},
        "by_category": {c: summarize(v) for c, v in by_cat.items()}
    }
