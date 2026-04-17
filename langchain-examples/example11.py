import os
import json
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

llm_key = os.getenv("OPENAI_API_KEY")

if not llm_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")


model = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=llm_key,
)



research_agent = create_agent(
    model=model,
    system_prompt="You are a research assistant. Provide detailed and informative explanations."
)

writer_agent = create_agent(
    model=model,
    system_prompt="You are a writer agent. Write content in a concise, clear, and well-structured manner."
)


@tool
def call_research_agent(query: str) -> str:
    """Call the research agent to gather detailed information on a topic."""
    print("1. Invoking Research Agent...\n")

    result = research_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })

    return result["messages"][-1].content


@tool
def call_writer_agent(query: str) -> str:
    """Call the writer agent to refine and summarize the content."""
    print("2. Invoking Writer Agent...\n")

    result = writer_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })

    return result["messages"][-1].content



class SupervisorResponse(BaseModel):
    topic: str = Field(..., description="User requested topic")
    research_summary: str = Field(..., description="Detailed research output")
    final_answer: str = Field(..., description="Final concise answer")




supervisor_agent = create_agent(
    model=model,
    tools=[call_research_agent, call_writer_agent],
    system_prompt="""
You are a supervisor agent coordinating between tools.

Workflow:
1. Use call_research_agent to gather detailed information.
2. Use call_writer_agent to summarize/refine the content.

IMPORTANT:
- Always call BOTH tools in sequence.
- First research, then writing.

Return ONLY valid JSON in this format:
{
    "topic": "<user topic>",
    "research_summary": "<detailed research output>",
    "final_answer": "<final refined answer>"
}
"""
)



query = "Research the benefits of protein and write a 2 paragraph summary."

response = supervisor_agent.invoke({
    "messages": [{"role": "user", "content": query}]
})


raw_output = response["messages"][-1].content

print("\nRaw Output:\n", raw_output)



try:
    parsed_output = SupervisorResponse(**json.loads(raw_output))

    print("\nParsed Output:\n")
    print("Topic:", parsed_output.topic)
    print("\nResearch Summary:\n", parsed_output.research_summary)
    print("\nFinal Answer:\n", parsed_output.final_answer)

except Exception as e:
    print("\nError parsing response:", e)