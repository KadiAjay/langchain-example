import os
from langchain.agents import create_agent
from langchain_core.tools import tool
from dotenv import load_dotenv
import requests
from tavily import TavilyClient
from pydantic import BaseModel, Field
from typing import List

load_dotenv()

client = TavilyClient(os.getenv("TAVILY_API_KEY"))

# ------------------ Models ------------------

class NewsArticle(BaseModel):
    """A single news article"""
    title: str = Field(description="Title of the article")
    source: str = Field(description="Publisher/source name")
    summary: str = Field(description="Brief summary of the article")
    url: str = Field(description="URL of the article (empty if unavailable)")


class AINewsResponse(BaseModel):
    """Structured response for AI news queries"""
    topic: str = Field(description="The news topic searched")
    articles: List[NewsArticle] = Field(description="List of relevant articles")
    overall_summary: str = Field(description="High-level summary of the topic")


# ------------------ Tools ------------------

@tool
def get_weather(location: str) -> str:
    """Fetch weather information for a given city."""
    
    api_key = os.getenv("OPENWEATHER_API_KEY")
    api_url = os.getenv("OPENWEATHER_API_URL")

    
    print(f"Fetching weather for {location}...")

    params = {
        "q": location,
        "appid": api_key,
        "units": "metric"
    }

    response = requests.get(api_url, params=params)

    if response.status_code != 200:
        return f"Error fetching weather for {location}"

    return str(response.json())


@tool
def websearch_tool(query: str) -> str:
    """Use this for general web search queries."""

    print(f"Performing web search for {query}...")
    
    response = client.search(
        query=query,
        search_depth="advanced"
    )
    return str(response)


# ------------------ Main ------------------

def main():
    topic = input("Enter the topic: ")
    
    if not topic:
        print("No topic provided")
        return

    print("Searching for news...\n")

    my_agent = create_agent(
        model="openai:gpt-4.1-mini",
        tools=[get_weather, websearch_tool],
        response_format=AINewsResponse,
        system_prompt="You are a helpful assistant that provides structured AI news summaries."
    )

    response = my_agent.invoke(
        {
            "messages": [
                {"role": "user", "content": f"Write a news article on {topic}"}
            ]
        }
    )

    # Extract structured response
    news: AINewsResponse = response["structured_response"]

    print(f"Topic: {news.topic}")
    print(f"\nOverall Summary: {news.overall_summary}")

    for article in news.articles:
        print(f"\n[{article.source}] {article.title}")
        print(f"{article.summary}")
        if article.url:
            print(f"URL: {article.url}")


# ------------------ Run ------------------

if __name__ == "__main__":
    main()