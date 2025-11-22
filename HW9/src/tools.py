import json
import re
from typing import Callable, List

from langchain.agents import Tool
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_core.language_models import BaseLanguageModel

from .chains import create_topic_decomposer_chain, create_subtopic_refinement_chain


def _wrap_topic_decompose(chain_invoke: Callable[[dict], dict], output_key: str) -> Callable[[str], str]:
    def _fn(input_text: str) -> str:
        # invoke 返回 dict，按 output_key 取值
        return chain_invoke({"topic": input_text})[output_key]
    return _fn


def _parse_subtopic_refiner_input(input_text: str) -> dict:
    """
    Preferred: JSON string {"subtopic": "...", "notes": "..."}
    Fallback: parse Subtopic: ... Notes: ...
    """
    try:
        data = json.loads(input_text)
        if isinstance(data, dict) and "subtopic" in data and "notes" in data:
            return {"subtopic": str(data["subtopic"]), "notes": str(data["notes"])}
    except Exception:
        pass

    pattern = re.compile(r"(?si)subtopic\s*:\s*(.+?)\s*notes\s*:\s*(.+)$")
    match = pattern.search(input_text)
    if match:
        return {"subtopic": match.group(1).strip(), "notes": match.group(2).strip()}

    return {"subtopic": "Unspecified subtopic", "notes": input_text.strip()}


def _wrap_subtopic_refiner(chain_invoke: Callable[[dict], dict], output_key: str) -> Callable[[str], str]:
    def _fn(input_text: str) -> str:
        parsed = _parse_subtopic_refiner_input(input_text)
        return chain_invoke(parsed)[output_key]
    return _fn


def _wrap_tavily(tavily: TavilySearchResults) -> Callable[[str], str]:
    def _fn(query: str) -> str:
        # Tavily 是 StructuredTool，显式用 query key
        return tavily.invoke({"query": query})
    return _fn


def build_research_tools(llm: BaseLanguageModel) -> List[Tool]:
    topic_chain = create_topic_decomposer_chain(llm)
    refiner_chain = create_subtopic_refinement_chain(llm)

    tavily = TavilySearchResults()

    tools: List[Tool] = [
        Tool(
            name="TopicSubtopicDecomposer",
            func=_wrap_topic_decompose(topic_chain.invoke, "subtopics_json"),
            description=(
                "Break down a broad research topic into 4–8 coherent subtopics. "
                "INPUT: a main topic string. "
                "OUTPUT: JSON array like "
                '[{{"subtopic": str, "why_important": str, "key_terms": [str, ...]}}].'
            ),
        ),
        Tool(
            name="WebSearchSubtopics",
            func=_wrap_tavily(tavily),
            description=(
                "Search the web for studies and references relevant to a subtopic. "
                "INPUT: a concise search query."
            ),
        ),
        Tool(
            name="SubtopicRefiner",
            func=_wrap_subtopic_refiner(refiner_chain.invoke, "refined_text"),
            description=(
                "Review and refine gathered notes for a specific subtopic. "
                "Preferred INPUT: JSON with keys 'subtopic' and 'notes'. Example: "
                '{{"subtopic": "Coordination protocols", "notes": "snippet1..."}} '
                "OUTPUT: ~120–180 words with inline citations and references."
            ),
        ),
    ]
    return tools
