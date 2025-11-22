import os
import json
from datetime import datetime
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

os.environ["OPENAI_API_KEY"]=''
os.environ["TAVILY_API_KEY"]=''

# Core LLM
llm = ChatOpenAI(model="gpt-4o", temperature=1)


# Defining a workflow for the Agent (Chains)

# Reflect and Refine answer
self_refinement_prompt = PromptTemplate(
    input_variables=["answer"],
    template="""
[Thought]:Evaluate the following answer for accuracy and completeness: {answer}
[Action]:Refine it if there is a problem
"""
)
self_refinement_chain = LLMChain(prompt=self_refinement_prompt, llm=llm, output_key="refined")

# Summarize refined answer
summary_prompt = PromptTemplate(
    input_variables=["refined"],
    template="""
[Action]:Summarize the following answer in a concise and clear way: {refined}
"""
)
summary_chain = LLMChain(prompt=summary_prompt, llm=llm, output_key="final_answer")


# Combine chains
multi_step_chain = SequentialChain(
      chains=[self_refinement_chain, summary_chain],
      input_variables=["answer"],
      output_variables=["final_answer"],
      verbose=True)



#Define Tools for the agent
tools = [
    Tool(
        name="Search",
        func=TavilySearchResults().run,
        description="Use this tool for answering questions with current, factual information from the web"
    ),
   Tool(
    name="FinalizeAnswer",
    func=multi_step_chain.run,
    description="Use this tool to refine, summarize, and finalize your answer to a question"
       )
]

#Define Memory for the Agent to keep conversations
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


#Define Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    max_iterations=50,
    max_execution_time=240,
    max_tokens=1000,
    verbose=True,
    handle_parsing_errors=True
)

#Query Agent (Assign a task to the Agent)
response = agent.run("Describe recent progress in agentic framework technologies 2025")
print(response)


#Save memory to a file (Persistent Memory)
with open("agent_memory.txt", "w") as f:
    for msg in memory.chat_memory.messages:
        f.write(f"{msg.content}\n")

