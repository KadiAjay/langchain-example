import os
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()
# os.environ["google_api_key"]="AIzaSyAD_Kx0qJwlE3asyRjuDjLY_mcnfQ9bqDI"


if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY  not found in environment")

writer_agent=create_agent(
    
    model="gpt-4.1-mini",
    system_prompt="""You are a professional content writer working for The Times of India. 
        Write clear, engaging, and well-structured articles in a journalistic tone."""
)
print("writer agent workded successfully")

editor_agent=create_agent(
    
    model="gpt-4.1-mini",
    system_prompt="""You are a professional content editor. Your role is to refine and improve content 
        written by the writer agent. Correct grammar, enhance clarity, improve flow, and 
        ensure consistency in tone and style"""
    
    
)
print("editor agent worked succesfully")

chain=writer_agent|editor_agent

response = chain.invoke(
    {
        "messages": [
            {"role": "user", "content": "write a news on trends in AI "}
        ]
    }
)



print(response["messages"][-1].content)