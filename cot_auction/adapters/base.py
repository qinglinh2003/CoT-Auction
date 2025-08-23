from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class GenConfig:
    k: int
    temperature: float
    max_steps: int

class ModelAdapter:
    def generate_chains(self, prompt: str, cfg: GenConfig) -> Dict[str, Any]:
        """Return: {"chains": [ { "final": str, "brief_rationale": str,
                                  "usage": {"in": int, "out": int}, "latency_s": float } ... ] }"""
        raise NotImplementedError
