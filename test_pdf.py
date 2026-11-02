
from core.pdf_processor import (
    extract_pdf_text,
    create_chunks
)

pdf_path = "data/documents/sample.pdf"

pages = extract_pdf_text(pdf_path)

chunks = create_chunks(pages)

print("Total pages:", len(pages))
print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks[:3]):

    print("\nCHUNK:", i + 1)
    print("PAGE:", chunk["page"])
    print(chunk["text"][:300])
