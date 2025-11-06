from typing import Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import LONGFORMER_MODEL_NAME


class LongformerSentimentAnalyzer:
    def __init__(self, model_name: str = None, max_length: int = 4096):
        self.model_name = model_name or LONGFORMER_MODEL_NAME
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        # try to read labels
        config = self.model.config
        self.id2label = getattr(config, "id2label", None)

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Run sentiment analysis on the given text and return label + score + raw logits.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_id = int(torch.argmax(probs, dim=-1).item())
        pred_score = float(probs[0, pred_id].item())
        if self.id2label is not None:
            pred_label = self.id2label.get(pred_id, f"LABEL_{pred_id}")
        else:
            # fallback
            # many sentiment models use 0=NEGATIVE, 1=POSITIVE
            mapping = {0: "NEGATIVE", 1: "POSITIVE"}
            pred_label = mapping.get(pred_id, f"LABEL_{pred_id}")

        return {
            "label": pred_label,
            "score": pred_score,
            "raw_logits": logits.tolist(),
            "model_name": self.model_name,
        }
