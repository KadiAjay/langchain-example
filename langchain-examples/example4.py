import os
from langchain.agents import create_agent
from dotenv import load_dotenv


load_dotenv()
os.environ["google_api_key"]="AIzaSyAD_Kx0qJwlE3asyRjuDjLY_mcnfQ9bqDI"
my_agent=create_agent(
    model="openai:gpt-4.1-mini",
    system_prompt=""" you are a linkedin post writer.
    you must create engeging and infomative posts.
    you should never work on anything else.
    if user asks you to do outside of this scope,
    politely decline."""
)

response=my_agent.invoke(
    {
        "messages":[
             {"role": "user", "content": "Write a linkedin post on trends in AI"}
        ]
    }

)
print(response["messages"][-1].content)

