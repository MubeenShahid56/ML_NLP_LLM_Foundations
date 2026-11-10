"""Week 03 - Phase 05: A small embedding-based semantic search system."""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def main():
    documents = [
        "Students can register courses through the student portal.",
        "The examination office publishes examination schedules.",
        "The finance office handles tuition fee issues.",
        "The library provides digital research resources.",
    ]

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Create document embeddings once and reuse them for each query.
    document_embeddings = model.encode(documents)

    queries = [
        "Where can I register my courses?",
        "Who publishes the exam timetable?",
        "Where can I get help with tuition fees?",
    ]

    for query in queries:
        query_embedding = model.encode([query])
        scores = cosine_similarity(
            query_embedding, document_embeddings
        )[0]

        ranked_results = sorted(
            zip(documents, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        print("\nQuery:", query)
        print("Ranked results:")
        for rank, (document, score) in enumerate(ranked_results, start=1):
            print(f"{rank}. {float(score):.4f} -> {document}")


if __name__ == "__main__":
    main()
