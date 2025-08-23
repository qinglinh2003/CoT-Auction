import os, glob, csv
from datetime import datetime
import matplotlib.pyplot as plt

def latest_csv(pattern="outputs/records_*.csv"):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError("No records_*.csv found in outputs/. Run eval_cli first.")
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def load_and_aggregate(path):
    by_k = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            k = int(r["k"])
            correct = int(r["correct"])
            cost = float(r["cost_usd"])
            if k not in by_k:
                by_k[k] = {"n":0, "sum_corr":0, "sum_cost":0.0}
            by_k[k]["n"] += 1
            by_k[k]["sum_corr"] += correct
            by_k[k]["sum_cost"] += cost
    points = []
    for k, v in by_k.items():
        n = max(1, v["n"])
        acc = v["sum_corr"] / n
        avg_cost = v["sum_cost"] / n
        points.append({"k": k, "acc": acc, "cost": avg_cost})
    points.sort(key=lambda x: x["cost"])
    return points

def pareto_frontier(points):
    frontier = []
    best_acc = -1.0
    for p in points:
        if p["acc"] > best_acc:
            frontier.append(p)
            best_acc = p["acc"]
    return frontier

def main():
    csv_path = latest_csv()
    pts = load_and_aggregate(csv_path)
    fr = pareto_frontier(pts)

    fig, ax = plt.subplots()
    xs = [p["cost"] for p in pts]
    ys = [p["acc"] for p in pts]
    ax.scatter(xs, ys)
    for p in pts:
        ax.annotate(f'k={p["k"]}', (p["cost"], p["acc"]), xytext=(3,3), textcoords="offset points", fontsize=9)

    if len(fr) >= 2:
        ax.plot([p["cost"] for p in fr], [p["acc"] for p in fr])

    ax.set_xlabel("Average cost (USD)")
    ax.set_ylabel("Accuracy")
    ax.set_title("CoT Auction | Accuracy–Cost by k")

    os.makedirs("outputs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_png = f"outputs/pareto_{ts}.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    print(f"[OK] Saved Pareto plot: {out_png}")
    print("[INFO] Points (by k):")
    for p in pts:
        print(f"  k={p['k']}: acc={p['acc']:.3f}, cost=${p['cost']:.5f}")

if __name__ == "__main__":
    main()
