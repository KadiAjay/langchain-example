import os
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_core.tools import tool
import requests

load_dotenv()

@tool
def get_weather(location: str) -> str:
    """
    Fetch weather information for a given city.
    """

    api_key = os.getenv("OPENWEATHER_API_KEY")
    api_url = os.getenv("OPENWEATHER_API_URL")

    print(f"Fetching weather information for {location}")

    params = {
        "q": location,
        "appid": api_key,
        "units": "metric"
    }

    response = requests.get(api_url, params=params)

    if response.status_code != 200:
        return f"Error fetching weather for {location}"

    data = response.json()
    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"Temperature: {temp}°C, Condition: {desc}"


my_agent = create_agent(
    model="google_genai:gemini-1.5-flash",  # safer default
    tools=[get_weather],
    system_prompt="You are a helpful assistant that provides weather information"
)

response = my_agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "What is the weather in New York now?"}
        ]
    },
    config={
        "tags": ["weather-agent", "example3"],
        "metadata": {
            "user_id": "user_001",
            "feature": "weather_lookup",
            "env": "dev"
        },
        "run_name": "weather_query_run"
    }
)

print(response["messages"][-1].text)