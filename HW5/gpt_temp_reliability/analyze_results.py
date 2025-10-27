"""
Analyze raw_runs.csv produced by run_experiment.py, with alias normalization.

Reliability metrics per temperature:
- Exact list stability rate
- Average pairwise Jaccard similarity
- Top-1 agreement rate
- Diversity: unique combo count / unique disease count
Outputs:
- outputs/summary_metrics.csv
- outputs/README_results.md

Notes:
- We normalize model outputs by:
  1) lowercasing
  2) trimming whitespace
  3) removing simple numbering/bullets and parentheses content
  4) splitting on commas / semicolons / newlines
  5) alias canonicalization (e.g., "septicemia" -> "sepsis")
"""

import os
import re
import pandas as pd
import numpy as np
from itertools import combinations

OUTPUT_DIR = "outputs"
RAW_CSV = os.path.join(OUTPUT_DIR, "raw_runs.csv")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "summary_metrics.csv")
REPORT_MD = os.path.join(OUTPUT_DIR, "README_results.md")

# ---------------------------------------------------------------------
# Alias map (extend as needed). LEFT: variant/synonym; RIGHT: canonical.
# Keep keys/values all lowercase.
ALIASES = {
    # Sepsis family
    "septicemia": "sepsis",
    "septic shock": "sepsis",
    "urosepsis": "sepsis",
    "bacteremia with sepsis": "sepsis",
    "severe sepsis": "sepsis",

    # Hypovolemia / dehydration cluster
    "volume depletion": "hypovolemia",
    "hypovolemic shock": "hypovolemia",
    "dehydration": "dehydration",  # keep as distinct if you prefer; or map to "hypovolemia"
    # If you want dehydration merged into hypovolemia, uncomment the next line:
    # "dehydration": "hypovolemia",

    # Adrenal insufficiency family
    "addison's disease": "adrenal insufficiency",
    "addison disease": "adrenal insufficiency",
    "primary adrenal insufficiency": "adrenal insufficiency",
    "adrenal crisis": "adrenal insufficiency",

    # Anemia variations
    "iron deficiency anemia": "anemia",
    "hemorrhagic anemia": "anemia",
    "acute blood loss anemia": "anemia",

    # Cardiogenic/infectious variants that sometimes appear
    "pneumonia with sepsis": "sepsis",
    "systemic inflammatory response syndrome": "sepsis",  # SIRS often conflated; adjust if needed

    # Orthostatic hypotension variants
    "postural hypotension": "orthostatic hypotension",
    "orthostatic hypotension (oh)": "orthostatic hypotension",

    # Thyrotoxicosis / hyperthyroid cluster (occasionally appears)
    "thyrotoxicosis": "hyperthyroidism",
    "thyroid storm": "hyperthyroidism",
}

# Optional: lightweight normalization replacements (common punctuation/typos)
REPLACEMENTS = [
    (r"\band\b", ","),              # "a, b and c" -> "a, b , c"
    (r"[•·▪●\-–—]\s*", ""),         # bullets/dashes
    (r"\s{2,}", " "),               # collapse spaces
]

SPLIT_PATTERN = re.compile(r"[,\n;]+")

def strip_numbering(s: str) -> str:
    # Remove leading numbering like "1.", "2)", "- ", etc.
    s = re.sub(r"^\s*(\d+[\.\)]\s*|\-\s*|\*\s*)", "", s)
    return s

def strip_parentheses(s: str) -> str:
    # Remove simple parenthetical notes, e.g., "sepsis (severe)" -> "sepsis"
    return re.sub(r"\s*\([^)]*\)\s*", " ", s)

def canon(s: str) -> str:
    """Lowercase, trim, remove extra spaces, apply alias map."""
    s = s.strip().lower()
    s = strip_numbering(s)
    s = strip_parentheses(s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return s
    return ALIASES.get(s, s)

def tokenize_to_items(text: str):
    """
    Split the raw model output into items by comma/semicolon/newline,
    clean each token, and keep up to 3 canonical disease names.
    """
    if not isinstance(text, str):
        return []

    # Basic replacements (bullets, "and" -> comma, etc.)
    t = text
    for pat, repl in REPLACEMENTS:
        t = re.sub(pat, repl, t, flags=re.IGNORECASE)

    # Split into tokens
    tokens = [tok.strip() for tok in SPLIT_PATTERN.split(t)]
    tokens = [tok for tok in tokens if tok]

    # Canonicalize & keep non-empty unique in order
    items = []
    seen = set()
    for tok in tokens:
        c = canon(tok)
        if c and c not in seen:
            items.append(c)
            seen.add(c)
        if len(items) >= 3:
            break
    return items

def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def analyze_temperature(df_temp: pd.DataFrame):
    df_temp = df_temp.copy()
    df_temp["diseases"] = df_temp["output_text"].apply(tokenize_to_items)
    df_temp["canonical_str"] = df_temp["diseases"].apply(lambda x: ",".join(x))

    # Exact list stability rate (most frequent 3-item combo)
    mode_combo = df_temp["canonical_str"].mode().iloc[0] if not df_temp.empty else ""
    exact_stability = (df_temp["canonical_str"] == mode_combo).mean()

    # Pairwise Jaccard mean across 5 runs
    idx = list(df_temp.index)
    pairs = list(combinations(idx, 2))
    j_scores = []
    for i, j in pairs:
        j_scores.append(jaccard(df_temp.at[i, "diseases"], df_temp.at[j, "diseases"]))
    j_mean = float(np.mean(j_scores)) if j_scores else 1.0

    # Top-1 agreement (after normalization)
    firsts = df_temp["diseases"].apply(lambda x: x[0] if len(x) > 0 else "")
    if not firsts.empty:
        first_mode = firsts.mode().iloc[0]
        top1_agree = (firsts == first_mode).mean()
    else:
        top1_agree = 0.0

    # Diversity indicators
    unique_combos = df_temp["canonical_str"].nunique()
    unique_diseases = len(set(d for lst in df_temp["diseases"] for d in lst))

    return {
        "n_runs": int(len(df_temp)),
        "exact_stability": round(float(exact_stability), 3),
        "pairwise_jaccard_mean": round(j_mean, 3),
        "top1_agreement": round(float(top1_agree), 3),
        "unique_combos": int(unique_combos),
        "unique_diseases": int(unique_diseases),
        "mode_combo": mode_combo,
    }

def main():
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"Not found: {RAW_CSV}. Run run_experiment.py first.")

    df = pd.read_csv(RAW_CSV)

    results = []
    for temp, sub in df.groupby("temperature"):
        m = analyze_temperature(sub)
        m["temperature"] = temp
        results.append(m)

    out = pd.DataFrame(results).sort_values("temperature")
    out.to_csv(SUMMARY_CSV, index=False, encoding="utf-8")

    # Write Markdown report
    lines = [
        "# Temperature Reliability Report (Alias-normalized)",
        "",
        "| temperature | runs | exact_stability | pairwise_jaccard_mean | top1_agreement | unique_combos | unique_diseases | mode_combo |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in out.iterrows():
        lines.append(
            f"| {row['temperature']} | {row['n_runs']} | {row['exact_stability']} | "
            f"{row['pairwise_jaccard_mean']} | {row['top1_agreement']} | "
            f"{row['unique_combos']} | {row['unique_diseases']} | {row['mode_combo']} |"
        )

    # Add a short note about aliasing for transparency
    lines += [
        "",
        "### Notes",
        "- Results above use alias normalization (e.g., `septicemia` → `sepsis`).",
        "- You can edit the `ALIASES` dict to adjust your canonicalization policy (e.g., merge `dehydration` into `hypovolemia`).",
    ]

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved metrics to {SUMMARY_CSV}")
    print(f"Saved markdown report to {REPORT_MD}")

if __name__ == "__main__":
    main()
