"""Week 03 - Phase 03: BERT/GPT-2 tokenization and token ID comparison.

The first run downloads pretrained tokenizer files from Hugging Face.
An internet connection is required for that first download.
"""

from transformers import AutoTokenizer


def show_tokenizer_details(model_name: str, text: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text, add_special_tokens=True)
    decoded_text = tokenizer.decode(token_ids)

    print("\nModel:", model_name)
    print("Tokens:", tokens)
    print("Token IDs:", token_ids)
    print("Decoded text:", decoded_text)
    print("Token count (including special tokens):", len(token_ids))

    if tokenizer.cls_token is not None:
        print("CLS token:", tokenizer.cls_token)
    if tokenizer.sep_token is not None:
        print("SEP token:", tokenizer.sep_token)

    return len(tokenizer.encode(text, add_special_tokens=False))


def main():
    text = "Large language models are powerful."
    print("Original text:", text)

    bert_count = show_tokenizer_details("bert-base-uncased", text)
    gpt2_count = show_tokenizer_details("gpt2", text)

    print("\nToken count comparison (without special tokens):")
    print("BERT:", bert_count)
    print("GPT-2:", gpt2_count)

    print(
        "\nNote: token counts can differ because tokenizers split text "
        "into subword units in different ways."
    )


if __name__ == "__main__":
    main()
