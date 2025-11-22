"""
Placeholders for fine-tuning support utilities.

Intended future functions:
- load and clean fine-tuning/validation datasets
- add instructions and convert to OpenAI JSONL format
- track and serialize hyperparameter configurations
"""

from __future__ import annotations
from typing import Iterable, Dict, Any
import json


def to_openai_jsonl(records: Iterable[Dict[str, Any]], path: str) -> None:
    """
    Write an iterable of dicts to a JSONL file for OpenAI fine-tuning.
    Each record should contain fields like: {"messages": [...]}.
    """
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


