from pypdf import PdfReader

def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

reader = PdfReader("doc.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text() or ""
text = " ".join(text.split())

chunks = split_text(text)
print("Кусков:", len(chunks))
print("--- Кусок 1 ---")
print(chunks[0])
print("--- Кусок 2 ---")
print(chunks[1])