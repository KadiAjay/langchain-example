import os
from langchain.chat_models import init_chat_model
os.environ["GOOGLE_API_KEY"] ="AIzaSyBsFFwAI4xrP7GbMA8gmTmZdbwXw_sEVKc"

model = init_chat_model("google_genai:gemini-3-pro-preview")
response = model.invoke("Why do parrots talk?")
print(response)