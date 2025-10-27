"""
Experiment: measure reliability of GPT-4 outputs under different temperatures.
We will repeat the same query 5 times each for temperature 0.1 and 1.4,
then save raw outputs for later analysis.
"""

import os
import time
import csv
from dataclasses import dataclass
from typing import List, Dict
import pandas as pd
from tqdm import tqdm

from openai import OpenAI

# ---------- Config ----------
MODEL_NAME = "gpt-4o-mini"  # or "gpt-4o" / "gpt-4.1" if your course specifies; keep consistent
TEMPERATURES = [0.1, 1.4]
REPEATS = 5
OUTPUT_DIR = "outputs"
RAW_CSV = os.path.join(OUTPUT_DIR, "raw_runs.csv")

# Prompt from assignment:
USER_PROMPT = (
    "a patient has a low blood pressure and a high heart rate. "
    "just give me names of three most probable diseases without any other word."
)

SYSTEM_MSG = (
    "You are a helpful assistant that STRICTLY outputs only disease names as a plain comma-separated list. "
    "No extra words, no explanations, no numbering."
)

# ---------- Data Model ----------
@dataclass
class RunRecord:
    model: str
    temperature: float
    run_index: int
    output_text: str
    ts: float

# ---------- Helpers ----------
def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_client() -> OpenAI:
    # API Key should be in env var: OPENAI_API_KEY
    return OpenAI()

def call_model(client: OpenAI, temperature: float) -> str:
    """
    Call the Chat Completions API and return the raw text string.
    We enforce minimal formatting via system prompt, but still need to sanitize later.
    """
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": USER_PROMPT},
        ],
    )
    text = resp.choices[0].message.content.strip()
    return text

def run_experiment() -> List[RunRecord]:
    client = get_client()
    records: List[RunRecord] = []

    for temp in TEMPERATURES:
        for i in tqdm(range(1, REPEATS + 1), desc=f"Temp {temp}"):
            try:
                txt = call_model(client, temp)
            except Exception as e:
                # Basic retry once on transient errors
                time.sleep(1.5)
                txt = call_model(client, temp)

            records.append(RunRecord(
                model=MODEL_NAME,
                temperature=temp,
                run_index=i,
                output_text=txt,
                ts=time.time()
            ))
            time.sleep(0.2)  # gentle pacing to avoid rate limits
    return records

def save_raw(records: List[RunRecord]):
    ensure_output_dir()
    rows = []
    for r in records:
        rows.append({
            "model": r.model,
            "temperature": r.temperature,
            "run_index": r.run_index,
            "output_text": r.output_text,
            "timestamp": r.ts
        })
    df = pd.DataFrame(rows)
    df.to_csv(RAW_CSV, index=False, encoding="utf-8")
    print(f"Saved raw runs to {RAW_CSV}")

if __name__ == "__main__":
    ensure_output_dir()
    data = run_experiment()
    save_raw(data)
