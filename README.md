# RAG PDF Bot

A command-line chatbot that answers questions about a PDF document using Retrieval-Augmented Generation (RAG). Instead of sending the whole document to the model, it finds the most relevant passages and answers based only on them.

## How it works

1. **Read** — extracts text from a PDF with `pypdf` and cleans up extra whitespace
2. **Chunk** — splits the text into overlapping chunks (300 characters, 50 overlap) so sentences aren't lost at chunk boundaries
3. **Embed** — converts every chunk into a 3072-dimensional vector using the Gemini embedding model
4. **Retrieve** — embeds the user's question and ranks chunks by cosine similarity
5. **Generate** — sends the top 3 chunks plus the question to a Gemini model, which answers using only that context and says so when the answer isn't there

## Project structure

| File | Purpose |
|---|---|
| `search.py` | Main app: indexing, retrieval and answering |
| `rag_utils.py` | Helper functions: PDF reading, chunking, cosine similarity |
| `hello.py`, `chat.py` | First steps: single request and simple chat with the model |
| `read_pdf.py`, `chunks.py`, `embed_test.py` | Experiments for each stage of the pipeline |
| `doc_chat.py` | Simple version that sends the whole document (no retrieval) |

## Setup

1. Clone the repo and create a virtual environment:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
2. Get a free API key at [Google AI Studio](https://aistudio.google.com) and create a `.env` file:
GEMINI_API_KEY=your_key_here
3. Put your PDF in the project folder as `doc.pdf`
4. Run:
python search.py
5. Or run the web version with PDF upload and chat history:
```
   streamlit run app.py
```
## What I learned

- **Chunking matters.** With 500-character chunks, a sentence about the team's location was buried in a chunk full of links and ranked poorly. Smaller 300-character chunks raised its similarity score and moved it to the top.
- **Cross-language retrieval is weaker.** Questions in Russian against an English document produced lower and less distinct similarity scores.
- **Retrieving several chunks adds robustness.** Even when the best chunk wasn't ranked first, it was in the top 3, so the model could still answer correctly.
- **Grounding prevents hallucination.** Telling the model to answer only from the context made it say "I don't know" for questions the document doesn't cover.
- **Real-world API limits.** I handled server overload (503) and free-tier quota errors (429) with `try/except` so the app keeps running.
- **LLMs have no memory.** Each API call is independent, so the app keeps chat history in `st.session_state` and sends the last messages with every prompt.
## Possible improvements

- Split chunks by sentences instead of characters
- Store embeddings in a vector database (Chroma / FAISS) instead of recomputing them on every run