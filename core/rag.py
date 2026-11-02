from core.embeddings import (
    generate_embeddings,
    generate_question_embedding,
)
from core.vector_store import VectorStore
from core.tutor import ask_tutor


class StudyRAG:

    def __init__(self):
        self.vector_store = VectorStore()
        self.document_loaded = False
        self.chunks = []

    def load_document(self, chunks):
        if not chunks:
            raise ValueError("No readable text found in PDF.")

        embeddings = generate_embeddings(chunks)
        self.vector_store.build_index(embeddings, chunks)

        self.chunks = chunks
        self.document_loaded = True

        return len(chunks)

    def retrieve_context(self, question, top_k=3):
        if not self.document_loaded:
            raise ValueError("Please upload a PDF first.")

        question_embedding = generate_question_embedding(question)
        results = self.vector_store.search(question_embedding, top_k=top_k)

        return results

    def ask_document(self, question, history=None):
        if not self.document_loaded:
            raise ValueError("Please upload a PDF first.")

        results = self.retrieve_context(question, top_k=3)

        context_blocks = []
        for result in results:
            context_blocks.append(
                f"Page {result['page']}:\n{result['text']}"
            )
        context = "\n\n".join(context_blocks)

        prompt = f"""
You are StudyMind AI, an academic study assistant.

Answer the student's question using the provided study material.

STUDY MATERIAL:
{context}

STUDENT QUESTION:
{question}

INSTRUCTIONS:
1. Explain concepts in simple language.
2. Provide examples when useful.
3. Use the study material as your primary source.
4. Do not invent facts or definitions.
5. If the answer is not available in the material, clearly say that.
6. Mention relevant source page numbers.
"""

        answer = ask_tutor(prompt, history=history)
        return answer, results