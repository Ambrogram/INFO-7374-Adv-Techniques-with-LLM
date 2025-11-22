import os
import json
from datetime import datetime

from .config import load_settings, get_llm, ensure_output_dir
from .agent import build_research_agent


def save_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    settings = load_settings()
    llm = get_llm(settings)

    agent, memory = build_research_agent(llm)

    query = "Write a literature review around 500 words with citation of studies on topic: multi-agent orchestration"

    # Run the agent
    final_answer = agent.invoke({"input": query})["output"]
    print("AGENT INPUT KEYS:", agent.input_keys)
    print("\n===== FINAL ANSWER =====\n")
    print(final_answer)
    print("\n========================\n")

    # Prepare outputs
    ensure_output_dir("outputs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    review_path = os.path.join(
        "outputs",
        f"literature_review_multi_agent_orchestration_{timestamp}.md",
    )
    save_text(review_path, final_answer)

    # Save conversation memory
    memory_lines = []
    for msg in memory.chat_memory.messages:
        role = getattr(msg, "type", "message")
        content = getattr(msg, "content", "")
        memory_lines.append(f"{role}: {content}")
    save_text(os.path.join("outputs", "agent_memory.txt"), "\n".join(memory_lines))


if __name__ == "__main__":
    main()


