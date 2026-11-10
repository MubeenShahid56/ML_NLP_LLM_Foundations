"""Week 03 - Phase 04: Sentence embeddings and semantic similarity."""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def main():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    sentences = [
        "How do I reset my password?",
        "I forgot my password.",
        "The weather is very hot today.",
    ]

    embeddings = model.encode(sentences)

    password_similarity = cosine_similarity(
        [embeddings[0]], [embeddings[1]]
    )[0][0]

    weather_similarity = cosine_similarity(
        [embeddings[0]], [embeddings[2]]
    )[0][0]

    print("Sentence 1:", sentences[0])
    print("Sentence 2:", sentences[1])
    print("Password similarity:", round(float(password_similarity), 4))

    print("\nSentence 1:", sentences[0])
    print("Sentence 3:", sentences[2])
    print("Weather similarity:", round(float(weather_similarity), 4))

    print(
        "\nInterpretation: a higher cosine similarity indicates that "
        "the sentence meanings are more similar in the embedding space."
    )


if __name__ == "__main__":
    main()
