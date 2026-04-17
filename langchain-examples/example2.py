import os
from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()

# os.environ["GOOGLE_API_KEY"] = "AIzaSyBuc0AMmj_q2OI4gihiU6zFR1j29oSEG8c"

my_agent = create_agent(
    model="google_genai:gemini-3.1-flash-lite-preview",
    system_prompt="You are a helpful assistant that provied the weather information"
)

response = my_agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "What is the weather in chennai"}
        ]
    }
)

print(response["messages"][-1].text)