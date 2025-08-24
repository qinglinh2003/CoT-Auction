import os, glob, json, csv, math, sys

def latest(globpat):
    files = glob.glob(globpat)
    if not files: return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def load_records(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for x in r:
            x["run_id"] = int(x.get("run_id", "1"))
            x["k"] = int(x["k"])
            x["correct"] = int(x["correct"])
            x["agreement"] = float(x["agreement"])
            x["cost_usd"] = float(x["cost_usd"])
            x["latency_s"] = float(x["latency_s"])
            rows.append(x)
    return rows

def load_details(path):
    with open(path, "r", encoding="utf-8") as f:
        ds = json.load(f)
    idx = {}
    for d in ds:
        key = (int(d.get("run_id",1)), d["qid"], int(d["k"]))
        idx[key] = d
    return ds, idx

def load_report(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def approx(a,b,eps=1e-6): return abs(a-b) <= eps

def main():
    rec = latest("outputs/records_*.csv")
    det = latest("outputs/details_*.json")
    rep = latest("outputs/report_*.json")
    if not rec or not det or not rep:
        print("[FAIL] Missing outputs. Need records_*.csv, details_*.json, report_*.json in outputs/")
        sys.exit(1)

    records = load_records(rec)
    details, dmap = load_details(det)
    report = load_report(rep)

    ok, err = 0, 0
    issues = []

    # Invariants per record
    for r in records:
        if r["k"] == 1 and not approx(r["agreement"], 1.0): 
            err += 1; issues.append(("agreement_k1", r)); continue
        if r["cost_usd"] < -1e-9 or r["latency_s"] < -1e-9:
            err += 1; issues.append(("nonneg_metrics", r)); continue

        key = (r["run_id"], r["qid"], r["k"])
        d = dmap.get(key)
        if not d:
            err += 1; issues.append(("missing_detail", r)); continue

        chains = d.get("chains", [])
        answers = d.get("answers", [])
        vote_hist = d.get("vote_hist", {})
        vote = d.get("vote", "")

        if len(chains) != r["k"]: 
            err += 1; issues.append(("chains_len", r)); continue
        if len(answers) != r["k"]:
            err += 1; issues.append(("answers_len", r)); continue
        if sum(vote_hist.values()) != r["k"]:
            err += 1; issues.append(("vote_hist_sum", r)); continue
        maj = max(vote_hist.items(), key=lambda kv: kv[1])[0] if vote_hist else ""
        if vote != maj:
            err += 1; issues.append(("vote_mismatch", r)); continue
        agr = (vote_hist[vote] / float(r["k"])) if vote and r["k"]>0 else 0.0
        if not approx(agr, r["agreement"]):
            err += 1; issues.append(("agreement_mismatch", r)); continue

        ok += 1

    # Report consistency
    acc_mean = sum(x["correct"] for x in records) / max(1,len(records))
    rep_acc = report.get("overall", {}).get("acc", None)
    if rep_acc is None or not approx(acc_mean, rep_acc, 1e-6):
        issues.append(("report_overall_acc", {"calc":acc_mean, "report":rep_acc})); err += 1
    else:
        ok += 1

    print(f"[SUMMARY] OK={ok}  ERR={err}  total_records={len(records)}")
    if issues:
        print("[SAMPLE_ISSUES]")
        for tag, obj in issues[:5]:
            print(tag, obj)
        sys.exit(2)
    else:
        print("[PASS] All invariants satisfied.")
        sys.exit(0)

if __name__ == "__main__":
    main()
