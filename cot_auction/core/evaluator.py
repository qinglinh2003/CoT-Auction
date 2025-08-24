from collections import Counter
from typing import List, Dict, Any
from cot_auction.core.parsing import parse_final
from cot_auction.core.metrics import judge_one, aggregate_metrics
from cot_auction.core.costing import to_token_cost
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

def _cfg_obj(k: int, strat: Dict[str, Any]):
    return type("Cfg", (), {
        "k": k,
        "temperature": strat["temperature"],
        "max_steps": strat["max_steps"],
    })()

def evaluate_dataset(adapter, dataset: List[Dict[str, Any]], strat: Dict[str, Any],
                     price_cfg: Dict[str, float], run_id: int | None = None):
    records: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    total = len(dataset) * len(strat["k_list"])
    pbar = tqdm(total=total, desc=f"Evaluating(run={run_id})", unit="call") if tqdm else None
    for qi, item in enumerate(dataset, 1):
        for k in strat["k_list"]:
            if pbar: pbar.set_postfix(q=qi, k=k)
            cfg = _cfg_obj(k, strat)
            gen = adapter.generate_chains(item["prompt"], cfg=cfg)
            finals = [parse_final(c) for c in gen["chains"]]
            vote_hist = Counter(finals)
            vote = vote_hist.most_common(1)[0][0] if finals else ""
            correct = judge_one(item, vote)
            agreement = vote_hist[vote] / max(1, len(finals)) if finals else 0.0
            token_in = sum(c.get("usage", {}).get("in", 0) for c in gen["chains"])
            token_out = sum(c.get("usage", {}).get("out", 0) for c in gen["chains"])
            cost = to_token_cost({"in": token_in, "out": token_out}, price_cfg)
            latency = gen.get("latency_s", 0.0)
            records.append({
                "run_id": run_id if run_id is not None else 1,
                "qid": item["id"], "category": item["category"], "k": k,
                "vote": vote, "correct": int(correct), "agreement": agreement,
                "token_input": token_in, "token_output": token_out,
                "cost_usd": cost["usd"], "latency_s": latency
            })
            chains_info = []
            for c, ans in zip(gen["chains"], finals):
                chains_info.append({
                    "answer": ans,
                    "brief": (c.get("brief_rationale") or "")[:280],
                    "usage": c.get("usage", {}),
                    "latency_s": c.get("latency_s", 0.0)
                })
            details.append({
                "run_id": run_id if run_id is not None else 1,
                "qid": item["id"], "category": item["category"], "k": k,
                "prompt": item["prompt"],
                "chains": chains_info,
                "answers": finals,
                "vote": vote,
                "vote_hist": dict(vote_hist),
                "agreement": agreement,
                "correct": int(correct)
            })
            if pbar: pbar.update(1)
    if pbar: pbar.close()
    report = aggregate_metrics(records)
    return records, report, details
