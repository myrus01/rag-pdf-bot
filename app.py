import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from rag_utils import read_pdf, split_text, similarity

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
Embed_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.1-flash-lite"

@st.cache_data
def index_document(text):
    chunks = split_text(text, chunk_size=300, overlap=50)
    result = client.models.embed_content(model=Embed_MODEL, contents=chunks)
    vectors = [e.values for e in result.embeddings]
    return chunks, vectors

st.title("RAG PDF Bot")
uploaded = st.file_uploader("Загрузите PDF файл", type="pdf")

if uploaded:
    text = read_pdf(uploaded)
    chunks, vectors = index_document(text)
    st.success(f"Документ проиндексирован!: {len(chunks)} кусков")

    if "message" not in st.session_state:
        st.session_state.message = []

    for msg in st.session_state.message:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    question = st.chat_input("Задайте вопрос по документу")
    if question:
        st.session_state.message.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        q_result = client.models.embed_content(model=Embed_MODEL, contents=question)
        q_vector = q_result.embeddings[0].values
        scores = [similarity(q_vector, v) for v in vectors]
        top = sorted(zip(scores, chunks), reverse=True)[:3]
        context = "\n\n".join(chunk for score, chunk in top)
        history = "\n".join(
            f"{m['role']}: {m['content']}" for m in st.session_state.message[-7:-1]
        )

        prompt = f"""Ты отвечаешь на вопросы по документу.
Используй контекст из документа и историю разговора.
Если ответа нет ни в контексте, ни в истории, так и скажи.
Отвечай на том же языке, на котором задан вопрос.

История разговора:
{history}

Контекст из документа:
{context}

Вопрос: {question}"""
        try:
            response =client.models.generate_content(model=CHAT_MODEL, contents=prompt)
            answer = response.text
        except Exception as e:
            answer = f"Ошибка: {e}. Попробуй задать вопрос ещё раз."

        st.session_state.message.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)