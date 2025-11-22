"""
Evaluate models on the test workbook for both phases:
 - BEFORE (baseline model): write responses to 'Before Fine-tuning' columns
 - AFTER (fine-tuned model): write responses to 'After Fine-tuning' columns

Usage examples (from project root):
  python -m src.run_baseline --phase before
  python -m src.run_baseline --phase after
  python -m src.run_baseline --phase after --model ft:... --outfile data/results/Test_Data_with_both_ft2.xlsx
"""
from __future__ import annotations

import argparse
import os
from typing import Tuple
from pathlib import Path

import pandas as pd

from config.settings import (
    PROCESSED_TEST_DATA_XLSX,
    BASELINE_RESULTS_XLSX,
    TEST_DATA_WITH_BOTH_FT1,
    DEFAULT_BASE_MODEL,
    FINE_TUNED_MODEL,
    RESULTS_DIR,
    get_openai_client,
)
from src.constants import (
    TARGET_SHEETS,
    COL_QUERY,
    COL_CORRECT_DIAG,
    COL_RESP_BEFORE,
    COL_RESULT_BEFORE,
    COL_PARSED_HELPER_BEFORE,
    COL_RESP_AFTER,
    COL_RESULT_AFTER,
    COL_PARSED_HELPER_AFTER,
)
from src.prompt_utils import (
    extract_diagnosis,
    normalize_diag,
    get_model_diag_candidates,
    drug_induced_match,
)


def call_model(prompt_text: str, model_name: str) -> str:
    client = get_openai_client()
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a medical diagnosis assistant. Follow the instructions precisely."},
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.2,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"API Error: {e}"


def evaluate_row(prompt_text: str, gold_diag: str, model_name: str) -> Tuple[str, str]:
    """
    Call the model and compute correctness.
    Returns (raw_response, parsed_diagnosis).
    """
    text = call_model(prompt_text, model_name)
    model_diag = extract_diagnosis(text)

    gold_norm = normalize_diag(gold_diag) if gold_diag else ""
    model_cands = get_model_diag_candidates(model_diag)

    if gold_norm and gold_norm in model_cands:
        result = "Correct"
    elif gold_norm and drug_induced_match(gold_norm, text):
        result = "Correct"
    else:
        result = "Wrong"

    return text, (model_diag or "")


def process_sheet_phase(df: pd.DataFrame, model_name: str, phase: str) -> pd.DataFrame:
    """
    Process a single sheet by calling the model and evaluating correctness.
    Assumes instruction is already prepended in COL_QUERY.
    Phase controls which columns to populate: BEFORE or AFTER.
    """
    if phase == "before":
        col_resp, col_result, col_parsed = COL_RESP_BEFORE, COL_RESULT_BEFORE, COL_PARSED_HELPER_BEFORE
    else:
        col_resp, col_result, col_parsed = COL_RESP_AFTER, COL_RESULT_AFTER, COL_PARSED_HELPER_AFTER

    # Ensure required cols exist
    for c in (COL_QUERY, COL_CORRECT_DIAG):
        if c not in df.columns:
            raise KeyError(f"Sheet is missing required column '{c}'")
    for c in (col_resp, col_result, col_parsed):
        if c not in df.columns:
            df[c] = pd.NA

    # Only process rows where the current phase response is empty
    mask = df[col_resp].isna() | (df[col_resp] == "")
    to_process = df[mask]
    if to_process.empty:
        return df

    for idx, row in to_process.iterrows():
        prompt_text = str(row[COL_QUERY]) if not pd.isna(row[COL_QUERY]) else ""
        gold_diag = str(row[COL_CORRECT_DIAG]) if not pd.isna(row[COL_CORRECT_DIAG]) else ""

        raw_resp, parsed_diag = evaluate_row(prompt_text, gold_diag, model_name)
        # recompute result using the same logic for safety
        gold_norm = normalize_diag(gold_diag) if gold_diag else ""
        model_cands = get_model_diag_candidates(parsed_diag)
        if gold_norm and gold_norm in model_cands:
            result = "Correct"
        elif gold_norm and drug_induced_match(gold_norm, raw_resp):
            result = "Correct"
        else:
            result = "Wrong"

        df.loc[idx, col_resp] = raw_resp
        df.loc[idx, col_result] = result
        df.loc[idx, col_parsed] = parsed_diag

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation for baseline (before) or fine-tuned (after) phases.")
    parser.add_argument("--phase", required=True, choices=["before", "after"], help="Evaluation phase to run.")
    parser.add_argument("--infile", type=str, help="Override input Excel path.")
    parser.add_argument("--outfile", type=str, help="Override output Excel path.")
    parser.add_argument("--model", type=str, help="Override model name.")
    args = parser.parse_args()

    # Select model
    if args.model:
        model_name = args.model
    elif args.phase == "before":
        model_name = DEFAULT_BASE_MODEL
    else:
        if not FINE_TUNED_MODEL:
            raise ValueError("FINE_TUNED_MODEL is not set. Provide via .env or --model.")
        model_name = FINE_TUNED_MODEL

    # Select IO files
    if args.phase == "before":
        infile = Path(args.infile) if args.infile else PROCESSED_TEST_DATA_XLSX
        outfile = Path(args.outfile) if args.outfile else BASELINE_RESULTS_XLSX
    else:
        infile = Path(args.infile) if args.infile else BASELINE_RESULTS_XLSX
        outfile = Path(args.outfile) if args.outfile else TEST_DATA_WITH_BOTH_FT1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not infile.exists():
        raise FileNotFoundError(f"Input file not found: {infile}")

    print(f"[INFO] Phase={args.phase} | Model={model_name}")
    print(f"[INFO] Input: {infile}")
    print(f"[INFO] Output: {outfile}")

    sheets: dict[str, pd.DataFrame] = pd.read_excel(infile, sheet_name=None)

    with pd.ExcelWriter(outfile, engine="openpyxl") as writer:
        for name, df in sheets.items():
            if name in TARGET_SHEETS:
                print(f"[INFO] Processing sheet: {name}")
                updated = process_sheet_phase(df.copy(), model_name, args.phase)
                updated.to_excel(writer, sheet_name=name, index=False)
            else:
                print(f"[INFO] Copying sheet as-is: {name}")
                df.to_excel(writer, sheet_name=name, index=False)

    print(f"[DONE] Saved results to: {outfile}")


if __name__ == "__main__":
    main()


