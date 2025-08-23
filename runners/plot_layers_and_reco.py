import os, glob, csv, argparse
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

def latest_csv(pattern="outputs/records_*.csv"):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError("No records_*.csv found in outputs/. Run eval_cli first.")
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def load_records(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for x in r:
            x["k"] = int(x["k"])
            x["correct"] = int(x["correct"])
            x["cost_usd"] = float(x["cost_usd"])
            rows.append(x)
    return rows

def aggregate_by(rows, key_fields=("k",), extra_filter=None):
    bucket = defaultdict(list)
    for r in rows:
        if extra_filter and not extra_filter(r): 
            continue
        key = tuple(r[k] for k in key_fields)
        bucket[key].append(r)
    out = []
    for key, lst in bucket.items():
        n = len(lst)
        acc = sum(x["correct"] for x in lst)/max(1,n)
        cost = sum(x["cost_usd"] for x in lst)/max(1,n)
        item = {k:v for k,v in zip(key_fields, key)}
        item.update(dict(n=n, acc=acc, cost=cost))
        out.append(item)
    out.sort(key=lambda d: (d["cost"], d.get("k", 0)))
    return out

def pareto_frontier(points):
    frontier = []
    best_acc = -1.0
    for p in sorted(points, key=lambda d: d["cost"]):
        if p["acc"] > best_acc:
            frontier.append(p)
            best_acc = p["acc"]
    return frontier

def reco_iso_accuracy(points, target):
    feasible = [p for p in points if p["acc"] >= target]
    if feasible:
        pick = min(feasible, key=lambda p: (p["cost"], p.get("k",0)))
        return dict(**pick, feasible=True)
    pick = max(points, key=lambda p: (p["acc"], -p.get("k",0)))
    return dict(**pick, feasible=False)

def reco_iso_cost(points, budget):
    feasible = [p for p in points if p["cost"] <= budget]
    if feasible:
        pick = max(feasible, key=lambda p: (p["acc"], -p.get("k",0)))
        return dict(**pick, feasible=True)
    pick = min(points, key=lambda p: p["cost"])
    return dict(**pick, feasible=False)

def plot_by_category(rows, out_dir, iso_acc=None, iso_cost=None):
    cats = sorted({r["category"] for r in rows})
    if not cats:
        raise ValueError("No categories found in records CSV.")
    n = len(cats)
    nrows = 2 if n>2 else 1
    ncols = 2 if n>1 else 1
    fig, axes = plt.subplots(nrows, ncols, squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    md_lines = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(out_dir, f"recommend_{ts}.md")

    for idx, cat in enumerate(cats):
        ax = axes_flat[idx]
        pts = aggregate_by(rows, key_fields=("k",), extra_filter=lambda r: r["category"]==cat)
        if not pts: 
            ax.set_visible(False); continue

        ax.scatter([p["cost"] for p in pts], [p["acc"] for p in pts])
        for p in pts:
            ax.annotate(f'k={p["k"]}', (p["cost"], p["acc"]), xytext=(3,3), textcoords="offset points", fontsize=8)
        fr = pareto_frontier(pts)
        if len(fr) >= 2:
            ax.plot([p["cost"] for p in fr], [p["acc"] for p in fr])

        ax.set_title(f"{cat}")
        ax.set_xlabel("Avg cost (USD)")
        ax.set_ylabel("Accuracy")

        if iso_acc is not None:
            ra = reco_iso_accuracy(pts, iso_acc)
            ax.scatter([ra["cost"]],[ra["acc"]], marker="x")
            md_lines.append(f"- [{cat}] Iso-Acc {iso_acc:.2f}: pick k={ra['k']} "
                            f"(acc={ra['acc']:.3f}, cost=${ra['cost']:.4f}, feasible={ra['feasible']})")
        if iso_cost is not None:
            rc = reco_iso_cost(pts, iso_cost)
            ax.scatter([rc["cost"]],[rc["acc"]], marker="s")
            md_lines.append(f"- [{cat}] Iso-Cost ${iso_cost:.3f}: pick k={rc['k']} "
                            f"(acc={rc['acc']:.3f}, cost=${rc['cost']:.4f}, feasible={rc['feasible']})")

    for j in range(idx+1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    out_png = os.path.join(out_dir, f"pareto_layers_{ts}.png")
    plt.savefig(out_png, dpi=160)

    if iso_acc is not None or iso_cost is not None:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# CoT Auction Recommendations\n\n")
            f.write("\n".join(md_lines) if md_lines else "_No recommendations_\n")

    print(f"[OK] Saved layered plot: {out_png}")
    if iso_acc is not None or iso_cost is not None:
        print(f"[OK] Saved recommendations: {md_path}")
        print("\n".join(md_lines))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="records CSV; default to latest in outputs/")
    ap.add_argument("--iso-acc", type=float, default=None, help="target accuracy for iso-accuracy recommendation")
    ap.add_argument("--iso-cost", type=float, default=None, help="budget (USD) for iso-cost recommendation")
    ap.add_argument("--out", default="outputs", help="output dir")
    args = ap.parse_args()

    csv_path = args.csv or latest_csv()
    rows = load_records(csv_path)
    os.makedirs(args.out, exist_ok=True)
    plot_by_category(rows, args.out, iso_acc=args.iso_acc, iso_cost=args.iso_cost)

if __name__ == "__main__":
    main()
