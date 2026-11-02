
import faiss
import numpy as np


class VectorStore:

    def __init__(self):
        self.index = None
        self.chunks = []

    def build_index(self, embeddings, chunks):
        """
        Creates a FAISS index from embeddings.
        """

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        dimension = embeddings.shape[1]

        # Create FAISS index
        self.index = faiss.IndexFlatIP(dimension)

        # Add embeddings to the index
        self.index.add(embeddings)

        self.chunks = chunks

        print("FAISS index created successfully!")
        print("Total vectors:", self.index.ntotal)

    def search(self, question_embedding, top_k=3):
        """
        Finds the most relevant chunks for a question.
        """

        if self.index is None:
            raise ValueError("Vector index has not been created.")

        question_embedding = np.asarray(
            question_embedding,
            dtype=np.float32
        )

        scores, indices = self.index.search(
            question_embedding,
            top_k
        )

        results = []

        for score, index in zip(scores[0], indices[0]):

            if index == -1:
                continue

            results.append({
                "page": self.chunks[index]["page"],
                "text": self.chunks[index]["text"],
                "score": float(score)
            })

        return results
