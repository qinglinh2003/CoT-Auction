import argparse, json, csv, os, yaml
from datetime import datetime

from cot_auction.adapters.dummy import DummyAdapter
from cot_auction.core.evaluator import evaluate_dataset

def load_dataset(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset = load_dataset(cfg["dataset_path"])
    strat = {
        "k_list": cfg["strategy"]["k_list"],
        "temperature": cfg["strategy"]["temperature"],
        "max_steps": cfg["strategy"]["max_steps"],
    }
    price_cfg = cfg["model"]["price"]

    adapter = DummyAdapter()  
    records, report = evaluate_dataset(adapter, dataset, strat, price_cfg)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"records_{ts}.csv")
    json_path = os.path.join(args.out_dir, f"report_{ts}.json")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[OK] Wrote CSV:   {csv_path}")
    print(f"[OK] Wrote JSON:  {json_path}")
    print(f"[INFO] Overall: acc={report['overall']['acc']:.3f}, avg_cost_usd={report['overall']['avg_cost_usd']:.4f}")

if __name__ == "__main__":
    main()
