import os
from langchain.agents import create_agent
from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv()
import requests
from tavily import TavilyClient

client = TavilyClient(os.getenv("TAVILY_API_KEY"))

@tool
def get_weather(location:str)->str:

    """
    A simple tool that simulates the feteching weather
     information for a given city.
    """

    api_key=os.getenv("OPENWEATHER_API_KEY")
    api_url=os.getenv("OPENWEATHER_API_URL")

    print(f"fetching weather information for {location}")

    params = {
        "q": location,
        "appid": api_key,
        "units": "metric"  
    }
    
    response = requests.get(api_url, params=params)
    
    if response.status_code != 200:
        return f"Error fetching weather for {location}"
    
    return response.json()

@tool
def websearch_tool(query:str)->str:
    """
    use this for general web search queries
    """
    print(f"performing webserach for {query}...")
    response=client.search(
    query=query,
    search_depth="advanced"
)
    return response

    
my_agent = create_agent(
    model="openai:gpt-4.1-mini",
    tools=[get_weather,websearch_tool],
    system_prompt="You are a helpful assistant that provied the weather information"
    
)

response = my_agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "What is the current weather in guwahati"}
        ]
    }
)

print(response["messages"][-1].text)

