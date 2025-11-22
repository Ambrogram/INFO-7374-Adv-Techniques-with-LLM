"""
Project settings and path configuration.

- Loads secrets and configuration from .env at the project root
- Exposes resolved data directories
- Provides an OpenAI client factory
"""
from pathlib import Path
import os

from dotenv import dotenv_values
from openai import OpenAI

# -----------------------------
# Project root & .env loading
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOTENV_PATH = PROJECT_ROOT / ".env"

# Read .env as a simple key-value dict (do NOT rely on system env overriding it)
dotenv_config = dotenv_values(DOTENV_PATH) if DOTENV_PATH.exists() else {}

# Always prefer the value from .env; fall back to real environment only if needed
OPENAI_API_KEY = dotenv_config.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")

# Model names
BASE_MODEL = dotenv_config.get("BASE_MODEL") or os.getenv("BASE_MODEL", "gpt-4o")
DEFAULT_BASE_MODEL = BASE_MODEL

FINE_TUNED_MODEL = (
    dotenv_config.get("FINE_TUNED_MODEL")
    or os.getenv("FINE_TUNED_MODEL", "")
)

# -----------------------------
# Data directories & paths
# -----------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

TEST_DATA_XLSX = RAW_DIR / "Test_Data.xlsx"
PROCESSED_TEST_DATA_XLSX = PROCESSED_DIR / "Test_Data_with_instr.xlsx"
BASELINE_RESULTS_XLSX = RESULTS_DIR / "Test_Data_before_ft_filled.xlsx"

# Additional result files for combined outputs (ft1/ft2)
TEST_DATA_WITH_BOTH_FT1 = RESULTS_DIR / "Test_Data_with_both_ft1.xlsx"
TEST_DATA_WITH_BOTH_FT2 = RESULTS_DIR / "Test_Data_with_both_ft2.xlsx"


def ensure_dirs() -> None:
    """
    Create data directories if they do not exist.
    """
    for path in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def get_openai_client() -> OpenAI:
    """
    Return an OpenAI client configured via .env.

    Raises:
        ValueError: if OPENAI_API_KEY is not set in .env or environment.
    """
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY not found. Please set it in .env at the project root."
        )
    return OpenAI(api_key=OPENAI_API_KEY)


# Ensure base directories exist upon import
ensure_dirs()
