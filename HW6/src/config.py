# src/config.py
import os
from dotenv import load_dotenv

# 加载 .env 里的 OPENAI_API_KEY
load_dotenv()

# 1) OpenAI key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# 2) 作业要求的两个 prompt
PROMPT_GOLD = "Give me a brief report on the return on investment in the gold market in 2022"
PROMPT_CRYPTO = "Give me a brief report on the return on investment in the cryptocurrency market in 2022"

# 3) Longformer 模型名字（需要的时候改这里）
LONGFORMER_MODEL_NAME = "allenai/longformer-base-4096"


# 4) 路径
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
REPORTS_DIR = "reports"

# 创建目录
for _d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, REPORTS_DIR]:
    os.makedirs(_d, exist_ok=True)
