import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

phrases = ["Сколько стоит Drex?", "Цены модели Drex за токен", "Где находится команда?"]
result = client.models.embed_content(model="gemini-embedding-001", contents=phrases)
vectors = [e.values for e in result.embeddings]

print("Длина вектора:", len(vectors[0]))

def similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b)

print("Фраза 1 и фраза 2:", similarity(vectors[0], vectors[1]))
print("Фраза 1 и фраза 3:", similarity(vectors[0], vectors[2]))