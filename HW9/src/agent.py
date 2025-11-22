from typing import Tuple

from langchain.agents import AgentType, initialize_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_core.language_models import BaseLanguageModel

from .tools import build_research_tools


def build_research_agent(llm: BaseLanguageModel) -> Tuple[AgentExecutor, ConversationBufferMemory]:
    """
    Build and return the research agent and its memory.
    - Tools: TopicSubtopicDecomposer, WebSearchSubtopics, SubtopicRefiner
    - LLM: provided ChatOpenAI instance (gpt-4o)
    - Memory: ConversationBufferMemory
    - Agent: ZERO_SHOT_REACT_DESCRIPTION with verbose logging
    - Limits: max_iterations=150, max_execution_time=480
    """
    tools = build_research_tools(llm)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        max_iterations=150,
        max_execution_time=480,
        verbose=True,
        handle_parsing_errors=True,
        early_stopping_method="generate",
    )
    return agent, memory


