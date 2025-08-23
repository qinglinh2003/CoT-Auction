from collections import Counter
from typing import List, Dict, Any
from cot_auction.core.parsing import parse_final
from cot_auction.core.metrics import judge_one, aggregate_metrics
from cot_auction.core.costing import to_token_cost

def _cfg_obj(k: int, strat: Dict[str, Any]):
    return type("Cfg", (), {
        "k": k,
        "temperature": strat["temperature"],
        "max_steps": strat["max_steps"],
    })()

def evaluate_dataset(adapter, dataset: List[Dict[str, Any]], strat: Dict[str, Any], price_cfg: Dict[str, float]):
    records = []
    for item in dataset:
        for k in strat["k_list"]:
            cfg = _cfg_obj(k, strat)
            gen = adapter.generate_chains(item["prompt"], cfg=cfg)

            finals = [parse_final(c) for c in gen["chains"]]
            vote = Counter(finals).most_common(1)[0][0] if finals else ""
            correct = judge_one(item, vote)
            agreement = finals.count(vote) / max(1, len(finals))

            token_in = sum(c["usage"]["in"] for c in gen["chains"])
            token_out = sum(c["usage"]["out"] for c in gen["chains"])
            cost = to_token_cost({"in": token_in, "out": token_out}, price_cfg)
            latency = gen.get("latency_s", 0.0)

            records.append({
                "qid": item["id"], "category": item["category"], "k": k,
                "vote": vote, "correct": int(correct), "agreement": agreement,
                "token_input": token_in, "token_output": token_out,
                "cost_usd": cost["usd"], "latency_s": latency
            })
    report = aggregate_metrics(records)
    return records, report
