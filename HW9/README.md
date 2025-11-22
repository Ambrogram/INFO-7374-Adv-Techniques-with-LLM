## Research Agent with LangChain (Topic Decomposition, Web Search, Refinement)

This project implements a research agent using LangChain and GPT-4o that:
- Breaks down a topic into coherent subtopics (Tool 1).
- Searches the web for each subtopic to gather studies and references (Tool 3).
- Reviews and refines information for each subtopic with inline citations (Tool 2).
- Synthesizes a ~500-word literature review with citations.

The agent is configured with memory and runs with generous iteration/time limits to fully explore and refine results.

### How Requirements Are Met
- 1) Topic Breakdown Chain (Tool 1): Implemented in `src/chains.py` as `create_topic_decomposer_chain`, exposed in `src/tools.py` as `TopicSubtopicDecomposer`. Produces 4–8 subtopics in machine-readable JSON-like structure.
- 2) Subtopic Review & Refinement Chain (Tool 2): Implemented in `src/chains.py` as `create_subtopic_refinement_chain`, exposed in `src/tools.py` as `SubtopicRefiner`. Produces structured explanations with inline citations.
- 3) Web Search Tool (Tool 3): Implemented in `src/tools.py` as `WebSearchSubtopics` using `TavilySearchResults`.
- 4) Research Agent: Defined in `src/agent.py` with `ChatOpenAI(model="gpt-4o", temperature=0.5)`, memory, and the three tools. Settings include `max_iterations=150`, `max_execution_time=480`, and `max_tokens=2000` (on the LLM).
- 5) Run Agent and Report: Entry point `src/run_agent.py` runs the agent on the exact query “Write a literature review around 500 words with citation of studies on topic: multi-agent orchestration”, prints the answer, and saves outputs to `outputs/`.

### Prerequisites
- Python 3.10+
- API keys:
  - `OPENAI_API_KEY`
  - `TAVILY_API_KEY`

You can store keys in environment variables or in a `.env` file at the project root:

```
OPENAI_API_KEY=your_openai_key_here
TAVILY_API_KEY=your_tavily_key_here
```

### Installation
1. Create and activate a virtual environment (recommended).
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Agent
You can run the agent in either of the following ways from the project root:
```
python -m src.run_agent
```
or
```
python src/run_agent.py
```

### What Happens When You Run It
- The agent will:
  - Decompose the topic into subtopics.
  - Search the web for each subtopic (studies, technical references).
  - Refine/structure the information with brief inline citations (e.g., `(Smith, 2022)`).
  - Produce a ~500-word literature review with a brief references list.
- Output files:
  - Final literature review: `outputs/literature_review_multi_agent_orchestration_<timestamp>.md`
  - Conversation memory: `outputs/agent_memory.txt`

### Interpreting the Output
- The main output is a concise literature review structured with:
  - Introductory context
  - Thematic body referencing specific subtopics
  - Brief conclusion
  - Inline citations in simple academic style
  - Optional References section (simple bibliography format)

Note: The agent uses live web search via Tavily. Ensure `TAVILY_API_KEY` is set. The references are grounded in the retrieved snippets but may require additional verification for formal academic use.

### Repository Structure
```
.
├─ requirements.txt
├─ README.md
├─ outputs/
│  └─ (generated files will be saved here)
└─ src/
   ├─ __init__.py
   ├─ config.py
   ├─ chains.py
   ├─ tools.py
   ├─ agent.py
   └─ run_agent.py
```

### Notes on Configuration
- Model: `gpt-4o`
- Temperature: `0.5` (balanced creativity and factuality)
- Max tokens: set on the LLM to `2000`
- Agent: `ZERO_SHOT_REACT_DESCRIPTION`
- Memory: `ConversationBufferMemory`
- Limits: `max_iterations=150`, `max_execution_time=480`

### Troubleshooting
- If you see authentication or rate-limit errors, verify `OPENAI_API_KEY` and `TAVILY_API_KEY`.
- If the agent stops early, try re-running; search APIs can have transient issues.
- Windows PowerShell: ensure you are in the project directory and that your virtual environment is activated before running.


