import re
from typing import Dict

def parse_final(chain: Dict) -> str:
    s = (chain.get("final") or "").strip()
    if s.lower().startswith("final:"):
        s = s.split(":", 1)[1].strip()
    return s

def normalize(s: str) -> str:
    return re.sub(r"\s+", "", s or "").strip().lower()
