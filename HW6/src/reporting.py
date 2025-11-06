# src/reporting.py
import os
from datetime import datetime
from typing import Dict, Any
from src.config import REPORTS_DIR


def write_sentiment_report(
    gold_text: str,
    gold_result: Dict[str, Any],
    crypto_text: str,
    crypto_result: Dict[str, Any],
    filename: str = "sentiment_report.md",
):
    """
    将 GPT-4 生成的两份报告和 Longformer 情感分析结果写入 Markdown 文件
    """
    report_path = os.path.join(REPORTS_DIR, filename)

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        f"# Sentiment Analysis Report",
        "",
        f"Generated at: {now}",
        "",
        "## 1. Gold (GPT-4 response)",
        "",
        "### GPT-4 Text",
        "",
        "```text",
        gold_text.strip(),
        "```",
        "",
        "### Sentiment Result (Longformer)",
        f"- Label: **{gold_result.get('label', 'N/A')}**",
        f"- Score: **{gold_result.get('score', 0):.4f}**",
        f"- Model: `{gold_result.get('model_name', 'unknown')}`",
        "",
        "## 2. Cryptocurrency (GPT-4 response)",
        "",
        "### GPT-4 Text",
        "",
        "```text",
        crypto_text.strip(),
        "```",
        "",
        "### Sentiment Result (Longformer)",
        f"- Label: **{crypto_result.get('label', 'N/A')}**",
        f"- Score: **{crypto_result.get('score', 0):.4f}**",
        f"- Model: `{crypto_result.get('model_name', 'unknown')}`",
        "",
        "## 3. Comparison / Notes",
        "",
        "- Both analyses used the same downstream model (Longformer).",
        "- Sentiment difference reflects GPT-4's tone about each market in 2022.",
    ]

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Report written to {report_path}")
