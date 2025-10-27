# Word2Vec on text8 — Analogy & Clustering Evaluation

## 1. Objective
The goal of this experiment is to evaluate how different Word2Vec hyper-parameters affect the model’s ability to capture semantic relationships. We train **16 Word2Vec models** on the **text8** corpus using all combinations of:

- `win_size ∈ {3, 7, 13, 25}`
- `vector_size ∈ {20, 70, 100, 300}`

Each model is assessed using:

1. **Analogy task**  
   - Compute: `Embedding("man") - Embedding("woman") + Embedding("daughter")`  
   - Use `similar_by_vector()` to retrieve the most similar word.

2. **K-Means clustering (K=3)**  
   - Cluster the embeddings of:  
     `['yen', 'yuan', 'spain', 'brazil', 'africa', 'asia']`  
   - Evaluate whether the model forms meaningful semantic groups (e.g., currencies vs. countries vs. continents).

---

## 2. Dataset
The **text8** corpus (from Wikipedia) is loaded via Gensim’s built-in dataset API, which supports automatic download and cached access.  
Only lowercase alphabetic tokens are preserved, making it a clean corpus for Word2Vec training.

---

## 3. Methods
- `gensim.models.Word2Vec` for model training
- `similar_by_vector()` for analogy evaluation
- `sklearn.cluster.KMeans` for semantic clustering
- Evaluation metrics stored in `result.csv`, including:
  - `analogy_top1`, `analogy_top1_sim`
  - `purity`, `silhouette`
  - Final ranking score `final_score`

---

## 4. Results Summary
All model outputs are recorded in `result.csv`. Based on the analogy and clustering performance, the model with:
win_size = 3
vector_size = 70


achieved the **highest final score**, with:

- Accurate analogy behavior (`man - woman + daughter →` *daughter/son family-semantic direction*)
- Perfect cluster purity (`purity = 1.0`)
- Highest overall semantic consistency among all 16 models

This indicates that a **small context window (3)** and a **moderate embedding dimension (70)** best preserved fine-grained semantic relationships in this experiment.

---

## 5. Conclusion
The evaluation shows that **win_size = 3, vector_size = 70** is the most effective hyper-parameter configuration for capturing conceptual relationships in the text8 corpus. Smaller windows tend to emphasize syntactic and local semantic relationships, enabling clearer analogy behavior and cleaner clustering boundaries. This configuration best balances semantic richness and noise control compared to larger windows or overly large vector dimensions.

---

## 6. Files Included
| File | Description |
|---|---|
| `w2v_grid2.py` | Training + evaluation script (16 models) |
| `result.csv` | Aggregated evaluation metrics |
| `README.md` | Analysis, methodology, and conclusion |

---

## 7. Reproducibility
To rerun the experiment:

```bash
pip install gensim sklearn pandas
python w2v_grid2.py


