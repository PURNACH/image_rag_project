import easyocr
import numpy as np


# Load OCR reader once
reader = easyocr.Reader(["en"], gpu=False)


def extract_text(processed_image):
    """
    Extract text from the processed image.
    """

    results = reader.readtext(
        processed_image,
        detail=1
    )

    extracted_text = []

    for result in results:
        if len(result) >= 2:
            text = result[1]

            if text.strip():
                extracted_text.append(text.strip())

    return "\n".join(extracted_text)