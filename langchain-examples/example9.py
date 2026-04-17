from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv


load_dotenv()


prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a seasoned salesperson experienced in writing persuasive product descriptions.
You write compelling product descriptions for e-commerce applications."""
     ),
    ("user",
     """Write a product description for the following product using these details:

Product Details: {query}

Include:
- Product Name
- Category
- Features
- Dimensions
- Price
- Release Date
"""
     )
])


llm_api_key = os.getenv("OPENAI_API_KEY")


model = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=llm_api_key
)


chain = prompt | model

response = chain.invoke({
    "query": "LG 4K TV XYZ1234, electronics, 120W sound, super clarity, dual glass, magic remote, 75 inches, ₹5,50,000, release in March 2025"
})


print(response.content)