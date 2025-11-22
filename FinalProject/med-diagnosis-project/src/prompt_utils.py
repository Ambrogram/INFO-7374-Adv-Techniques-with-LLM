"""
Prompt and parsing utilities for the medical diagnosis workflow.
"""
from __future__ import annotations
import re
import string
from typing import List

from .constants import INSTR


def prepend_instruction(text: str) -> str:
    """
    Prepend the global instruction to a patient condition string.
    Guards against double-prepending by checking the beginning.
    """
    if not text:
        return INSTR
    stripped = str(text).strip()
    if stripped.startswith("Suppose that you are a medical diagnosis assistant"):
        return stripped
    return INSTR + stripped


def extract_diagnosis(text: str) -> str:
    """
    Extract the first 'Diagnosis: ...' line from a possibly multi-line LLM response.

    Returns the content after 'Diagnosis:' (tolerates some punctuation variants)
    and stops at the first newline or at the start of known sections like
    'Probability:', 'Rationale:', etc.
    """
    if not text:
        return ""

    text = str(text).strip()
    match = re.search(r"diagnosis\s*[:：-]\s*(.*)", text, flags=re.IGNORECASE)
    if not match:
        return text.strip()

    rest = match.group(1).strip()
    rest = rest.split("\n")[0].strip()

    cut_keywords = [
        "probability:",
        "probabilities:",
        "urgency:",
        "rationale:",
        "reason:",
        "treatment:",
        "treatments:",
    ]
    for kw in cut_keywords:
        idx = rest.lower().find(kw)
        if idx != -1:
            rest = rest[:idx].strip()
            break

    return rest.strip(" .:-")


def normalize_diag(name: str) -> str:
    """
    Normalize a diagnosis string to make comparison more robust.

    Operations:
    - lowercasing
    - trimming spaces
    - removing trailing punctuation
    - collapsing multiple spaces
    """
    if not name:
        return ""
    name = str(name).lower().strip()
    name = name.strip(string.punctuation + " ")
    name = " ".join(name.split())
    return name


def get_model_diag_candidates(model_diag: str) -> List[str]:
    """
    Turn a model's diagnosis line into a list of candidate diagnoses.

    Example:
        "Chronic Obstructive Pulmonary Disease (COPD)"
    ->  ["chronic obstructive pulmonary disease (copd)",
         "chronic obstructive pulmonary disease",
         "copd"]
    """
    if not model_diag:
        return []

    model_diag = model_diag.strip()
    candidates: List[str] = []

    full_norm = normalize_diag(model_diag)
    if full_norm:
        candidates.append(full_norm)

    if "(" in model_diag and ")" in model_diag:
        before = model_diag.split("(", 1)[0].strip()
        inside = model_diag.split("(", 1)[1].split(")", 1)[0].strip()

        before_norm = normalize_diag(before)
        inside_norm = normalize_diag(inside)

        if before_norm and before_norm not in candidates:
            candidates.append(before_norm)
        if inside_norm and inside_norm not in candidates:
            candidates.append(inside_norm)

    return candidates


def drug_induced_match(gold_norm: str, model_full_text: str) -> bool:
    """
    Heuristic matcher for labels like '<drug>-induced sexual dysfunction'. We check that
    both the drug name and the condition substring exist in the full model output.
    """
    if not gold_norm:
        return False

    gold_lower = gold_norm.lower()
    model_lower = model_full_text.lower()

    drug_name = None
    if "-induced" in gold_lower:
        drug_name = gold_lower.split("-induced", 1)[0].strip()

    condition = None
    if "sexual dysfunction" in gold_lower:
        condition = "sexual dysfunction"

    if drug_name and condition:
        if drug_name in model_lower and condition in model_lower:
            return True

    return False


