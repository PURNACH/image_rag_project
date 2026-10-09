import re


def clean_text(text):
    """
    Clean OCR extracted text.
    """

    if not text:
        return ""

    # Remove extra spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive new lines
    text = re.sub(r"\n+", "\n", text)

    # Remove unwanted characters
    text = re.sub(r"[^\w\s.,!?():/%+\-&@#]", "", text)

    return text.strip()


def chunk_text(text, chunk_size=500, overlap=100):
    """
    Split text into overlapping chunks.
    """

    if not text:
        return []

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


def save_extracted_text(text, filename="extracted_text.txt"):
    """
    Save extracted text to a file.
    """

    with open(filename, "w", encoding="utf-8") as file:
        file.write(text)

    return filename