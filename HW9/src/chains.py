from typing import Dict

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.language_models import BaseLanguageModel


def create_topic_decomposer_chain(llm: BaseLanguageModel) -> LLMChain:
    """
    Create a chain that breaks a main topic into 4–8 subtopics with rationale.
    Output is a machine-readable JSON-like structure.
    """
    prompt = PromptTemplate(
        input_variables=["topic"],
        template=(
            "You are a research planner.\n"
            "Decompose the MAIN TOPIC into 4–8 specific, logically coherent subtopics.\n"
            "Return ONLY a compact JSON array of objects with this schema:\n"
            '[{{"subtopic": str, "why_important": str, "key_terms": [str, ...]}}]\n\n'
            "Constraints:\n"
            "- No prose outside JSON\n"
            "- Subtopics should be non-overlapping and cover core aspects\n"
            "- Keep text concise and factual\n\n"
            "MAIN TOPIC: {topic}\n"
        ),
    )
    return LLMChain(prompt=prompt, llm=llm, output_key="subtopics_json")


def create_subtopic_refinement_chain(llm: BaseLanguageModel) -> LLMChain:
    """
    Create a chain that reviews and refines information for a subtopic.
    Inputs: subtopic (str), notes (str of raw snippets, quotes, URLs).
    Output: coherent explanation with inline citations and a tiny references list.
    """
    prompt = PromptTemplate(
        input_variables=["subtopic", "notes"],
        template=(
            "You are an academic writing assistant.\n"
            "Task: Review the NOTES for the SUBTOPIC, ensure relevance, remove redundancy, "
            "and produce a coherent explanation with brief inline citations and a simple references list.\n\n"
            "SUBTOPIC: {subtopic}\n"
            "NOTES:\n"
            "{notes}\n\n"
            "Requirements:\n"
            "- Length: ~120–180 words\n"
            "- Address relevance, coherence, and basic factual consistency\n"
            "- Use inline citations like (Author, Year) linked to the sources in NOTES\n"
            "- End with 'References' listing 2–4 plausible works in simple bibliography format\n"
            "- Output plain text only\n"
        ),
    )
    return LLMChain(prompt=prompt, llm=llm, output_key="refined_text")


def run_topic_decomposer(llm: BaseLanguageModel, topic: str) -> str:
    """Helper to run the topic decomposer chain directly."""
    chain = create_topic_decomposer_chain(llm)
    return chain.run({"topic": topic})


def run_subtopic_refinement(llm: BaseLanguageModel, subtopic: str, notes: str) -> str:
    """Helper to run the subtopic refinement chain directly."""
    chain = create_subtopic_refinement_chain(llm)
    return chain.run({"subtopic": subtopic, "notes": notes})


