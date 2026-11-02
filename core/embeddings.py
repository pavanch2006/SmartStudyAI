
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("Loading embedding model...")

embedding_model = SentenceTransformer(MODEL_NAME)


def generate_embeddings(text_chunks):
    """
    Converts text chunks into numerical embeddings.
    """

    texts = [chunk["text"] for chunk in text_chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings


def generate_question_embedding(question):
    """
    Converts a student's question into an embedding.
    """

    embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embedding
