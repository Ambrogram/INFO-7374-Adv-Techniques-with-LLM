import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


@dataclass
class Settings:
    """Centralized configuration for the research agent."""
    openai_api_key: str
    tavily_api_key: Optional[str]
    model_name: str = "gpt-4o"
    temperature: float = 0.5
    max_tokens: int = 2000


def load_settings() -> Settings:
    """
    Load environment variables and return a Settings object.
    Looks for .env and process environment variables.
    """
    load_dotenv(override=False)
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    tavily_api_key = os.getenv("TAVILY_API_KEY", "")

    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Set it in your environment or .env file.")

    # Tavily key is recommended for web search; warn if missing but don't hard fail.
    if not tavily_api_key:
        print("Warning: TAVILY_API_KEY not set. Web search tool may not function.")

    return Settings(
        openai_api_key=openai_api_key,
        tavily_api_key=tavily_api_key or None,
    )


def get_llm(settings: Settings) -> ChatOpenAI:
    """
    Construct and return the core ChatOpenAI LLM.
    max_tokens is set here to meet the assignment's requirement.
    """
    # The ChatOpenAI client reads OPENAI_API_KEY from env by default.
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    return ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


def ensure_output_dir(path: str) -> None:
    """Create output directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


