import os

from src.config import (
    PROMPT_GOLD,
    PROMPT_CRYPTO,
    RAW_DIR,
)
from src.gpt_client import generate_report
from src.sentiment_analyzer import LongformerSentimentAnalyzer
from src.reporting import write_sentiment_report


def run_pipeline():
    # 1. call GPT for gold
    print("[1/4] Calling GPT-4 for GOLD report ...")
    gold_text = generate_report(PROMPT_GOLD)
    gold_path = os.path.join(RAW_DIR, "gold_report.txt")
    with open(gold_path, "w", encoding="utf-8") as f:
        f.write(gold_text)
    print(f"    saved GPT gold report to {gold_path}")

    # 2. call GPT for crypto
    print("[2/4] Calling GPT-4 for CRYPTO report ...")
    crypto_text = generate_report(PROMPT_CRYPTO)
    crypto_path = os.path.join(RAW_DIR, "crypto_report.txt")
    with open(crypto_path, "w", encoding="utf-8") as f:
        f.write(crypto_text)
    print(f"    saved GPT crypto report to {crypto_path}")

    # 3. sentiment analysis with Longformer
    print("[3/4] Running Longformer sentiment analysis ...")
    analyzer = LongformerSentimentAnalyzer()

    gold_result = analyzer.analyze(gold_text)
    crypto_result = analyzer.analyze(crypto_text)

    print(f"    GOLD sentiment: {gold_result['label']} ({gold_result['score']:.4f})")
    print(f"    CRYPTO sentiment: {crypto_result['label']} ({crypto_result['score']:.4f})")

    # 4. write report
    print("[4/4] Writing report ...")
    write_sentiment_report(
        gold_text=gold_text,
        gold_result=gold_result,
        crypto_text=crypto_text,
        crypto_result=crypto_result,
    )
    print("Done. See reports/sentiment_report.md")
