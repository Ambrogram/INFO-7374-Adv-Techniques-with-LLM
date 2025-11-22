"""
Build OpenAI fine-tuning JSONL files from FineTuning and Validation CSVs.

Assignment Step C.3: prepend INSTR to queries (user messages).
Assignment Step C.4: convert to JSONL in OpenAI format.

Notes:
- Only prepend INSTR to the Query (user) side.
- Do not modify the assistant Response content.
- Skip rows with empty/NaN Query.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from config.settings import RAW_DIR, PROCESSED_DIR
from src.constants import INSTR


def _ensure_processed_dir() -> None:
    """
    Ensure the processed directory exists.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    """
    Ensure required columns exist in the DataFrame.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Input CSV is missing required columns: {missing}")


def df_to_openai_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert a DataFrame with 'Query' and 'Response' columns to OpenAI messages records.

    - Prepends INSTR to the user content: INSTR + Query
    - Keeps assistant Response unchanged
    - Skips rows where Query is empty/NaN
    """
    records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        query_val = row.get("Query", None)
        if pd.isna(query_val) or str(query_val).strip() == "":
            # Skip rows without user input
            continue

        user_text = f"{INSTR}{str(query_val).strip()}"

        resp_val = row.get("Response", "")
        assistant_text = "" if pd.isna(resp_val) else str(resp_val)

        records.append(
            {
                "messages": [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ]
            }
        )

    return records


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    """
    Write a list of dicts as JSON Lines with UTF-8 encoding.
    """
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_file(input_csv: Path, output_jsonl: Path) -> int:
    """
    Read an input CSV, validate schema, convert to JSONL records,
    and write to output path. Returns number of examples written.
    """
    df = pd.read_csv(input_csv)
    _validate_columns(df, ["Query", "Response"])
    records = df_to_openai_records(df)
    write_jsonl(output_jsonl, records)
    return len(records)


def main() -> None:
    _ensure_processed_dir()

    fine_tune_csv = RAW_DIR / "FineTuning_Data.csv"
    valid_csv = RAW_DIR / "Validation_Data.csv"

    fine_tune_out = PROCESSED_DIR / "FineTuning_Data.jsonl"
    valid_out = PROCESSED_DIR / "Validation_Data.jsonl"

    num_ft = build_file(fine_tune_csv, fine_tune_out) if fine_tune_csv.exists() else 0
    print(f"[INFO] Wrote {num_ft} examples to {fine_tune_out}") if num_ft else print(
        f"[WARN] Input not found or empty: {fine_tune_csv}"
    )

    num_val = build_file(valid_csv, valid_out) if valid_csv.exists() else 0
    print(f"[INFO] Wrote {num_val} examples to {valid_out}") if num_val else print(
        f"[WARN] Input not found or empty: {valid_csv}"
    )


if __name__ == "__main__":
    main()


