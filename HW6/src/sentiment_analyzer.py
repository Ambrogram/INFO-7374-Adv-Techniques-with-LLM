# src/sentiment_analyzer.py

from typing import Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import LONGFORMER_MODEL_NAME


class LongformerSentimentAnalyzer:
    def __init__(self, model_name: str = None, max_length: int = 4096):
        """
        model_name: 要加载的长文本模型名称，默认从 config 里读
        max_length: tokenizer 截断长度，Longformer 一般是 4096
        """
        self.model_name = model_name or LONGFORMER_MODEL_NAME
        self.max_length = max_length

        # 加载 tokenizer 和模型
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        # 尝试从模型配置里读出 id2label（有些模型会自带，比如 0 -> NEGATIVE, 1 -> POSITIVE）
        config = self.model.config
        self.id2label = getattr(config, "id2label", None)

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        对一段文本做情感分析，返回 label + score + 原始 logits
        """
        # 编码
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        # 前向推理
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits  # [1, num_labels]
        probs = torch.softmax(logits, dim=-1)
        pred_id = int(torch.argmax(probs, dim=-1).item())
        pred_score = float(probs[0, pred_id].item())

        # 我们自己的兜底标签：大部分二分类情感模型都是 0=NEGATIVE, 1=POSITIVE
        fallback = {0: "NEGATIVE", 1: "POSITIVE"}

        if self.id2label is not None:
            # 模型自己有 id2label，就优先用它
            pred_label = self.id2label.get(
                pred_id,
                fallback.get(pred_id, f"LABEL_{pred_id}")
            )
        else:
            # 模型没有 id2label，就用我们自己的兜底；如果类别不是 0/1，就退回 LABEL_x
            pred_label = fallback.get(pred_id, f"LABEL_{pred_id}")

        return {
            "label": pred_label,
            "score": pred_score,
            "raw_logits": logits.tolist(),
            "model_name": self.model_name,
        }
