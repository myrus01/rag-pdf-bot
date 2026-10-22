import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

while True:
    question = input("Ты: ")
    if question == "exit":
        break
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=question
    )
    print("AI:", response.text)