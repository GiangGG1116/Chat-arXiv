import pdfplumber
from googletrans import Translator
from typing import List

def read_pdf_text(filepath: str) -> str:
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def chunk_text(text: str, max_chars: int = 500) -> List[str]:
    lines = text.split("\n")
    chunks = []
    current = ""
    for line in lines:
        if len(current) + len(line) < max_chars:
            current += line + "\n"
        else:
            chunks.append(current.strip())
            current = line + "\n"
    if current:
        chunks.append(current.strip())
    return chunks

def translate_document(filepath: str) -> List[str]:
    raw_text = read_pdf_text(filepath)
    chunks = chunk_text(raw_text)
    translator = Translator()
    translated = []
    for chunk in chunks:
        try:
            vi = translator.translate(chunk, src="en", dest="vi").text
            translated.append(vi)
        except Exception as e:
            translated.append(f"[Lỗi dịch đoạn]: {str(e)}")
    return translated
