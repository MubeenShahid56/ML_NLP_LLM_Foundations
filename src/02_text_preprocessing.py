"""Week 03 - Phase 02: NLP text preprocessing."""

import re


def clean_text(text: str) -> str:
    """Remove extra whitespace and punctuation, and lowercase text."""
    text = text.strip()
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def main():
    example_sentences = [
        " I NEED HELP!!! ",
        "Natural   Language Processing is useful.",
        "Machine Learning, NLP, and LLMs!",
    ]

    for sentence in example_sentences:
        print("Original:", repr(sentence))
        print("Cleaned: ", clean_text(sentence))
        print()


if __name__ == "__main__":
    main()
