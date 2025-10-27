# -*- coding: utf-8 -*-
"""
w2v_grid.py — Train 16 Word2Vec models on text8, evaluate analogy & clustering, pick the best.
"""

import os
import itertools
import warnings
warnings.filterwarnings("ignore")

from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

import gensim
import gensim.downloader as api
from gensim.models import Word2Vec

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from itertools import permutations

# ----------------------------
# 0) Config
# ----------------------------
WIN_SIZES = [3, 7, 13, 25]
VECTOR_SIZES = [20, 70, 100, 300]

MIN_COUNT = 5
SG = 0  # 0: CBOW, 1: Skip-gram (作业未指定，保持默认CBOW即可)
WORKERS = max(1, os.cpu_count() - 1)
EPOCHS = 5  # text8 不算很大，5~10轮都行，出成绩更平滑可调高

RESULTS_CSV = "results.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# 评估词与“黄金标准”聚类
ANALOGY_POS = ("man", "daughter")   # man - woman + daughter
ANALOGY_NEG = ("woman",)
ANALOGY_TARGET = "son"
CLUSTER_TOKENS = ['yen','yuan','spain','brazil','africa','asia']
TRUE_CLUSTERS = [
    set(['yen','yuan']),       # currencies
    set(['spain','brazil']),   # countries
    set(['africa','asia'])     # continents
]

def load_text8_sentences():
    """
    使用 gensim 的 downloader 加载 text8，返回句子迭代器（list of list of tokens）。
    """
    print("Loading text8 via gensim.downloader ...")
    dataset = api.load("text8")   # 返回一个可迭代对象，每项是一个已分词的句子
    # 为了可多次遍历，装入内存（text8约100MB，可接受；也可直接用迭代器训练一次）
    sentences = [list(sent) for sent in dataset]
    print(f"Loaded {len(sentences)} 'sentences' (chunks) from text8.")
    return sentences

def word_in_vocab(model, w: str) -> bool:
    return w in model.wv.key_to_index

def safe_vector(model, w: str):
    return model.wv[w]

def compute_analogy_score(model) -> Dict[str, object]:
    """
    计算 Transform = v('man') - v('woman'), q = Transform + v('daughter')
    返回：
      - top10 list
      - son_rank (1-based, None)
      - son_cos (float or None)
      - score_analogy ∈ [0,1]
    """
    words_needed = set([*ANALOGY_POS, *ANALOGY_NEG, ANALOGY_TARGET])
    if not all(word_in_vocab(model, w) for w in words_needed):
        return {
            "top10": [],
            "son_rank": None,
            "son_cos": None,
            "score_analogy": 0.0
        }

    v_man = safe_vector(model, ANALOGY_POS[0])
    v_daughter = safe_vector(model, ANALOGY_POS[1])
    v_woman = safe_vector(model, ANALOGY_NEG[0])

    transform = v_man - v_woman
    query = transform + v_daughter

    topn = model.wv.similar_by_vector(query, topn=10)
    # topn: list of (word, cosine)
    top10_words = [w for w,_ in topn]

    # 记录 'son' 的排名与相似度
    son_rank = None
    son_cos = None
    for idx, (w, c) in enumerate(topn, start=1):
        if w == ANALOGY_TARGET:
            son_rank = idx
            son_cos = c
            break

    # 名次转分：Top-10 内 (11-rank)/10, 否则 0
    if son_rank is not None:
        score_rank = (11 - son_rank) / 10.0
    else:
        score_rank = 0.0

    # 可选：把相似度（若存在）再线性融合一点点（占比不大，避免不稳定）
    if son_cos is not None:
        score_analogy = 0.9 * score_rank + 0.1 * max(0.0, min(1.0, son_cos))
    else:
        score_analogy = score_rank

    return {
        "top10": topn,
        "son_rank": son_rank,
        "son_cos": son_cos,
        "score_analogy": float(score_analogy)
    }

def best_purity_mapping(pred_labels: List[int], tokens: List[str]) -> Tuple[float, Dict[int, set]]:
    """
    计算在 BEST 簇-真类匹配下的 purity。
    由于只有 3 簇，遍历 3! = 6 种映射足够。
    返回 (purity, 映射后的真类集 for 每个簇标签)
    """
    # 将预测结果按簇分组
    groups = {}
    for t, lab in zip(tokens, pred_labels):
        groups.setdefault(lab, []).append(t)

    # 对簇标签的所有排列进行匹配，最大化命中数
    labels = list(groups.keys())
    K = len(labels)
    assert K == 3, "Expect K=3."

    best_hits = -1
    best_perm = None
    # 所有 K! 种映射到 TRUE_CLUSTERS 的排列
    for perm in permutations(range(3), 3):
        # perm[i] 表示预测簇 labels[i] 对应 TRUE_CLUSTERS[perm[i]]
        hits = 0
        for i, lab in enumerate(labels):
            pred_set = set(groups[lab])
            true_set = TRUE_CLUSTERS[perm[i]]
            hits += len(pred_set & true_set)
        if hits > best_hits:
            best_hits = hits
            best_perm = perm

    purity = best_hits / len(tokens)
    # 组装映射信息（可返回用于报告）
    mapping = { labels[i]: TRUE_CLUSTERS[best_perm[i]] for i in range(3) }
    return purity, mapping

def compute_clustering_scores(model) -> Dict[str, object]:
    """
    对指定 6 个词做 KMeans(K=3)，评估 purity 与 silhouette。
    若缺词，返回 0 分并说明缺少的词。
    """
    missing = [w for w in CLUSTER_TOKENS if not word_in_vocab(model, w)]
    if missing:
        return {
            "missing": missing,
            "purity": 0.0,
            "silhouette": None,
            "labels": None
        }

    X = np.vstack([safe_vector(model, w) for w in CLUSTER_TOKENS])
    kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)
    labels = kmeans.fit_predict(X)

    try:
        sil = float(silhouette_score(X, labels))
    except Exception:
        sil = None

    purity, mapping = best_purity_mapping(labels.tolist(), CLUSTER_TOKENS)
    return {
        "missing": [],
        "purity": float(purity),
        "silhouette": sil,
        "labels": labels.tolist(),  # 与 CLUSTER_TOKENS 顺序一致
        "mapping": mapping
    }

def train_one(sentences, win_size: int, vector_size: int) -> Word2Vec:
    print(f"Training Word2Vec(win={win_size}, dim={vector_size}) ...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=win_size,
        min_count=MIN_COUNT,
        sg=SG,
        workers=WORKERS,
        epochs=EPOCHS
    )
    out = os.path.join(MODELS_DIR, f"w2v_win{win_size}_dim{vector_size}.model")
    model.save(out)
    print(f"Saved model -> {out}")
    return model

def main():
    sentences = load_text8_sentences()

    rows = []
    details_for_report = {}  # 保存更详细的top10/labels以便写报告

    for w, d in itertools.product(WIN_SIZES, VECTOR_SIZES):
        model = train_one(sentences, w, d)

        # 1) 类比评估
        analog = compute_analogy_score(model)
        # 记录 top1 词与其相似度，便于肉眼审查
        top1_word, top1_sim = (None, None)
        if analog["top10"]:
            top1_word, top1_sim = analog["top10"][0]

        # 2) 聚类评估
        clus = compute_clustering_scores(model)

        # 3) 综合分
        score_analogy = analog["score_analogy"]
        purity = clus["purity"]
        final_score = 0.6 * score_analogy + 0.4 * purity

        rows.append({
            "win_size": w,
            "vector_size": d,
            "analogy_top1": top1_word,
            "analogy_top1_sim": top1_sim,
            "son_rank": analog["son_rank"],
            "son_cos": analog["son_cos"],
            "score_analogy": score_analogy,
            "purity": purity,
            "silhouette": clus["silhouette"],
            "final_score": final_score,
            "cluster_labels_(yen,yuan,spain,brazil,africa,asia)": None if clus["labels"] is None else ",".join(map(str, clus["labels"])),
            "cluster_missing": None if not clus["missing"] else ",".join(clus["missing"])
        })

        # 为报告保存详细对象（可序列化也行）
        details_for_report[(w, d)] = {
            "analogy_top10": analog["top10"],
            "cluster_labels": clus["labels"],
            "cluster_mapping": clus.get("mapping"),
            "cluster_missing": clus["missing"]
        }

    df = pd.DataFrame(rows).sort_values(by=["final_score","purity","score_analogy"], ascending=False)
    df.to_csv(RESULTS_CSV, index=False)
    print("\n=== Summary (Top 10 by final_score) ===")
    print(df.head(10).to_string(index=False))
    best = df.iloc[0]
    print("\n=== Best Hyper-parameters ===")
    print(f"win_size={int(best['win_size'])}, vector_size={int(best['vector_size'])}")
    print(f"final_score={best['final_score']:.3f} | purity={best['purity']:.3f} | analogy_score={best['score_analogy']:.3f}")
    print(f"analogy_top1={best['analogy_top1']} (sim={best['analogy_top1_sim']}) | son_rank={best['son_rank']} | son_cos={best['son_cos']}")
    print(f"labels(yen,yuan,spain,brazil,africa,asia)={best['cluster_labels_(yen,yuan,spain,brazil,africa,asia)']}")
    print(f"missing_for_cluster={best['cluster_missing']}")
    print(f"\nFull table saved to: {RESULTS_CSV}")

if __name__ == "__main__":
    main()
