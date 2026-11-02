import pymupdf
import pytesseract
from PIL import Image
import io

# Tesseract installation path
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def extract_pdf_text(pdf_path):

    document = pymupdf.open(pdf_path)
    extracted_pages = []

    for page_number, page in enumerate(document):

        # Convert PDF page into an image
        pix = page.get_pixmap(dpi=200)

        image = Image.open(
            io.BytesIO(pix.tobytes("png"))
        )

        # Extract text using OCR
        text = pytesseract.image_to_string(
            image,
            lang="eng"
        )

        if text.strip():
            extracted_pages.append({
                "page": page_number + 1,
                "text": text
            })

    document.close()

    return extracted_pages


def create_chunks(pages, chunk_size=1000, overlap=150):

    chunks = []

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):

            end = start + chunk_size
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append({
                    "page": page["page"],
                    "text": chunk_text
                })

            start += chunk_size - overlap

    return chunks