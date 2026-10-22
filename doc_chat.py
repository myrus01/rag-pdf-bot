import os
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

reader = PdfReader("doc.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text() or ""
text = " ".join(text.split())

while True:
    question = input("You: ")
    if question == "exit":
        break
    prompt = f"""Ответь на вопрос, используя только информацию из документа. Если ответа нет в документе, ответь "Извините, я не знаю ответа на этот вопрос.
Документ: {text}
Вопрос: {question}"""
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt
    )
    print("AI:", response.text)