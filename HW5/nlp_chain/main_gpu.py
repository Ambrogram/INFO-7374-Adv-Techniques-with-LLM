import os, re, json, time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    pipeline,
)

# French input text to be processed
INPUT_TEXT_FR = (
    "J'apprécie le sérieux avec lequel ce restaurant prend les allergies alimentaires. "
    "En tant que personne allergique aux noix, je me sentais complètement en sécurité en dînant ici. "
    "De plus, leurs options sans gluten et végétaliennes ont été une agréable surprise. "
    "Fortement recommandé à toute personne ayant des restrictions alimentaires Community Verified icon"
)

def clean_text(text: str) -> str:
    """
    Remove unwanted trailing labels or noise from the original French text.
    Here we remove the 'Community Verified icon' suffix that is not part of the content.
    """
    text = re.sub(r"\bCommunity Verified icon\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def assert_gpu():
    """
    Ensure that a CUDA-capable GPU is available. If not, raise an error.
    This prevents accidentally running on CPU when the intent is to use GPU acceleration.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU was NOT detected. Make sure GPU PyTorch (cu121) is installed, "
            "your NVIDIA driver is working, and your virtual environment is Python 3.11 with GPU support."
        )

def build_translation_pipeline(device):
    """
    Build a translation pipeline (French → English) using a Seq2Seq translation model.
    The model is loaded in float16 (FP16) to reduce memory usage and speed up GPU inference.
    """
    name = "Helsinki-NLP/opus-mt-fr-en"
    tok  = AutoTokenizer.from_pretrained(name)
    mdl  = AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype=torch.float16).to(device)
    return pipeline("translation", model=mdl, tokenizer=tok, device=device.index), name

def build_summarization_pipeline(device):
    """
    Build an English summarization pipeline using a BART-based Seq2Seq model.
    Also loaded in FP16 and moved to GPU for faster inference.
    """
    name = "facebook/bart-large-cnn"
    tok  = AutoTokenizer.from_pretrained(name)
    mdl  = AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype=torch.float16).to(device)
    return pipeline("summarization", model=mdl, tokenizer=tok, device=device.index), name

def build_sentiment_pipeline(device):
    """
    Build an English sentiment-analysis pipeline (positive/negative).
    Uses a lightweight DistilBERT classifier. FP16 + GPU for speed and consistency.
    """
    name = "distilbert-base-uncased-finetuned-sst-2-english"
    tok  = AutoTokenizer.from_pretrained(name)
    mdl  = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.float16).to(device)
    return pipeline("sentiment-analysis", model=mdl, tokenizer=tok, device=device.index), name

def main():
    # Ensure output directory exists
    os.makedirs("outputs", exist_ok=True)

    # Validate GPU availability
    assert_gpu()

    # Explicitly use the first CUDA GPU
    device = torch.device("cuda:0")

    # Clean the original French text (mostly removes suffix noise)
    text_fr = clean_text(INPUT_TEXT_FR)

    # ---- Stage 1: Translation (FR → EN) ----
    t0 = time.time()
    translator, trans_name = build_translation_pipeline(device)
    with torch.inference_mode():  # no gradient, faster
        translation = translator(text_fr, truncation=True)[0]["translation_text"].strip()
    t1 = time.time()

    # ---- Stage 2: Summarization (EN → EN summary) ----
    summarizer, sum_name = build_summarization_pipeline(device)
    with torch.inference_mode():
        summary = summarizer(translation, max_length=60, min_length=15, do_sample=False)[0]["summary_text"].strip()
    t2 = time.time()

    # ---- Stage 3: Sentiment Analysis (summary → sentiment label + score) ----
    sentiment_pipe, senti_name = build_sentiment_pipeline(device)
    with torch.inference_mode():
        senti = sentiment_pipe(summary)[0]
    t3 = time.time()

    # Structure all results into a clean JSON payload
    result = {
        "python_version": f"{torch.__version__} (torch)",
        "device": torch.cuda.get_device_name(0),
        "dtype": "float16",
        "models": {
            "translation": trans_name,
            "summarization": sum_name,
            "sentiment": senti_name
        },
        "input_french": text_fr,
        "translated_english": translation,
        "summary_english": summary,
        "sentiment_label": senti.get("label"),
        "sentiment_score": round(float(senti.get("score", 0.0)), 4),
        "latency_seconds": {
            "translate": round(t1 - t0, 3),
            "summarize": round(t2 - t1, 3),
            "sentiment": round(t3 - t2, 3),
            "total": round(t3 - t0, 3)
        }
    }

    # Write results to JSON file
    with open("outputs/result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Print for immediate readability in console
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
