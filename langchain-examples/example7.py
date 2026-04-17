import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch


load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY  not found in environment")

tavily_tool=TavilySearch(
    max_result=10,
    topic="general",
    search_depth="advanced",
)

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a professional content writer working for The Times of India. 
        Write clear, engaging, and well-structured articles in a journalistic tone.
        use the search_tool to get the latest information about the topic.
        
"""),
    ("user", "{topic}")
])

editor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a professional content editor. Your role is to refine and improve content 
        written by the writer agent. Correct grammar, enhance clarity, improve flow, and 
        ensure consistency in tone and style
"""),
    ("user", "{article}")
])



writer_agent = create_agent(
    model="gpt-4.1-mini",
    tools=[tavily_tool],
)

editor_agent = create_agent(
    model="gpt-4.1-mini",
    
)



def run_chain(user_input):
    
   
    writer_messages = writer_prompt.format_messages(topic=user_input)
    
    writer_response = writer_agent.invoke({
        "messages": writer_messages
    })
    
    article = writer_response["messages"][-1].content


    editor_messages = editor_prompt.format_messages(article=article)
    
    editor_response = editor_agent.invoke({
        "messages": editor_messages
    })

    final_output = editor_response["messages"][-1].content

    return final_output


if __name__ == "__main__":
    topic = "give me the ipl schedule 2026"
    
    result = run_chain(topic)
    

    print(result)