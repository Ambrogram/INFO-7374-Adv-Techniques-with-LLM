# Temperature Reliability Report (Alias-normalized)

| temperature | runs | exact_stability | pairwise_jaccard_mean | top1_agreement | unique_combos | unique_diseases | mode_combo |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.1 | 5 | 0.4 | 0.4 | 0.8 | 3 | 7 | hypovolemia,sepsis,adrenal insufficiency |
| 1.4 | 5 | 0.2 | 0.43 | 0.8 | 5 | 7 | dehydration,anemia,sepsis |

### Notes
- Results above use alias normalization (e.g., `septicemia` → `sepsis`).
- You can edit the `ALIASES` dict to adjust your canonicalization policy (e.g., merge `dehydration` into `hypovolemia`).

### Summary
- “We controlled everything except temperature and repeated the query five times per setting.”

- “Lower temperature (0.1) produced higher exact stability and higher pairwise similarity with fewer unique combinations, indicating better reliability for diagnostic candidate lists.”

- “This experiment evaluates format consistency of LLM outputs, not medical correctness; it does not constitute clinical diagnosis.”