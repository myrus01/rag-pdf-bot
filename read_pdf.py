from pypdf import PdfReader

reader = PdfReader("doc.pdf")
print("Pages:", len(reader.pages))

text = ""
for page in reader.pages:
    text += page.extract_text() or ""

print("Symbols:", len(text))
print(text[:500])