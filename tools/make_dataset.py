import json, random
from datetime import date, timedelta

random.seed(42)

def arith_items(n=15):
    items = []
    for i in range(1, n+1):
        a = random.randint(12, 49)
        b = random.randint(12, 49)
        ans = a * b
        items.append({
            "id": f"arith_{i:03d}",
            "category": "arithmetic",
            "prompt": f"What is {a}*{b}? Give a number.",
            "answer": str(ans)
        })
    return items

def unit_items(n=15):
    items = []
    for i in range(1, n+1):
        kind = random.choice(["km2m","m2cm","min2s","h2min"])
        if kind == "km2m":
            x = random.randint(1, 35) + random.choice([0, 5]) * 0.1
            ans = int(round(x * 1000))
            prompt = f"Convert {x} km to meters. Give a number (no unit)."
        elif kind == "m2cm":
            x = random.randint(2, 90)
            ans = x * 100
            prompt = f"Convert {x} m to centimeters. Give a number (no unit)."
        elif kind == "min2s":
            x = random.randint(2, 90)
            ans = x * 60
            prompt = f"Convert {x} minutes to seconds. Give a number (no unit)."
        else:  # h2min
            x = random.randint(2, 48)
            ans = x * 60
            prompt = f"Convert {x} hours to minutes. Give a number (no unit)."
        items.append({
            "id": f"unit_{i:03d}",
            "category": "unit",
            "prompt": prompt,
            "answer": str(ans)
        })
    return items

def date_items(n=15):
    items = []
    base = date(2022, 1, 1)
    for i in range(1, n+1):
        d = base + timedelta(days=random.randint(0, 900))
        k = random.randint(3, 60)
        ans = d + timedelta(days=k)
        items.append({
            "id": f"date_{i:03d}",
            "category": "date",
            "prompt": f"What is {d.isoformat()} plus {k} days? Answer in YYYY-MM-DD.",
            "answer": ans.isoformat()
        })
    return items

def logic_items(n=15):
    items = []
    for i in range(1, n+1):
        kind = random.choice(["transitive","parity","subset"])
        if kind == "transitive":
            prompt = "If Alice is taller than Bob and Bob is taller than Carl, is Alice taller than Carl? Answer YES or NO."
            ans = "YES"
        elif kind == "parity":
            x = random.randint(3, 50)
            prompt = f"If n is even, is n*{x*2} always even? Answer YES or NO."
            ans = "YES"
        else:
            prompt = "All bloops are fuzzles. All fuzzles are snorps. Are all bloops snorps? Answer YES or NO."
            ans = "YES"
        items.append({
            "id": f"logic_{i:03d}",
            "category": "logic",
            "prompt": prompt,
            "answer": ans
        })
    return items

def main():
    data = []
    data += arith_items(15)
    data += unit_items(15)
    data += date_items(15)
    data += logic_items(15)
    with open("data/mini_plus.jsonl", "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[OK] Wrote {len(data)} items to data/mini_plus.jsonl")

if __name__ == "__main__":
    main()
