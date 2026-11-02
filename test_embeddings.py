
from core.pdf_processor import (
    extract_pdf_text,
    create_chunks
)

from core.embeddings import (
    generate_embeddings,
    generate_question_embedding
)

from core.vector_store import VectorStore


# STEP 1: Extract PDF text
pdf_path = "data/documents/sample.pdf"

pages = extract_pdf_text(pdf_path)

chunks = create_chunks(pages)

print("\nTotal chunks:", len(chunks))


# STEP 2: Generate embeddings
embeddings = generate_embeddings(chunks)

print("\nEmbedding shape:", embeddings.shape)


# STEP 3: Create FAISS vector database
vector_store = VectorStore()

vector_store.build_index(
    embeddings,
    chunks
)


# STEP 4: Ask a test question
question = "What is a Context Free Grammar?"

question_embedding = generate_question_embedding(question)


# STEP 5: Retrieve relevant chunks
results = vector_store.search(
    question_embedding,
    top_k=3
)


# STEP 6: Display results
print("\nRETRIEVED RESULTS")

for i, result in enumerate(results):

    print("\nResult:", i + 1)
    print("Page:", result["page"])
    print("Similarity:", result["score"])
    print("Text:")
    print(result["text"][:500])
