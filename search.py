import os
from dotenv import load_dotenv
from google import genai
from rag_utils import read_pdf, split_text, similarity

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
EMBED_MODEL = "gemini-embedding-001"

chunks = split_text(read_pdf("doc.pdf"), chunk_size=300, overlap=50)
result = client.models.embed_content(model=EMBED_MODEL, contents=chunks)
chunk_vectors = [e.values for e in result.embeddings]
print("Проиндексировано кусков:", len(chunks))

while True:
    question = input("You: ")
    if question in ["выход", "exit"]:
        break

    # 1. Поиск: находим 3 самых похожих куска
    q_result = client.models.embed_content(model=EMBED_MODEL, contents=question)
    q_vector = q_result.embeddings[0].values
    scores = [similarity(q_vector, v) for v in chunk_vectors]
    top = sorted(zip(scores, chunks), reverse=True)[:3]

    # 2. Склеиваем найденные куски в контекст
    context = "\n\n".join(chunk for score, chunk in top)

    # 3. Отправляем модели контекст и вопрос
    prompt = f"""Ответь на вопрос, используя только этот контекст.
Если в контексте нет ответа, так и скажи.
Отвечай на том же языке, на котором задан вопрос.

Контекст:
{context}

Вопрос: {question}"""
    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=prompt
        )
        print("AI:", response.text)
    except Exception as e:
        print("Ошибка:", e)
        print("Попробуй задать вопрос ещё раз.")