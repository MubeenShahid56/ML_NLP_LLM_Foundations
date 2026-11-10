# Week 03 — Machine Learning, NLP & LLM Foundations

This folder follows the **Week 03 Checklist** and uses the supplied **Week 03.docx** as the code reference.

## Folder structure

```text
Week_03_ML_NLP_LLM_Foundations/
├── data/
├── evidence/
│   └── README.md
├── src/
│   ├── 01_text_classification.py
│   ├── 02_text_preprocessing.py
│   ├── 03_tokenization.py
│   ├── 04_embeddings_similarity.py
│   └── 05_semantic_search.py
├── requirements.txt
├── WEEK_03_CHECKLIST.md
└── README.md
```

## 1. Requirements

- Python 3.10 or newer
- Internet connection for the first download of pretrained BERT, GPT-2, and sentence-transformer model files

## 2. Setup (Chromebook/Linux)

Open the Linux Terminal and move into this project folder. If the folder is in Downloads, first copy it into your Linux files.

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install the packages:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## 3. Run the programs

Run these commands from the main project folder:

```bash
python3 src/01_text_classification.py
python3 src/02_text_preprocessing.py
python3 src/03_tokenization.py
python3 src/04_embeddings_similarity.py
python3 src/05_semantic_search.py
```

The tokenizer and sentence-transformer programs download pretrained model files the first time they run. This can take time and requires an internet connection.

## 4. What each program does

| File | Work |
|---|---|
| `01_text_classification.py` | Creates a small labeled text dataset, converts text with CountVectorizer, splits data, trains Naive Bayes, predicts new text, and prints accuracy and a classification report. |
| `02_text_preprocessing.py` | Cleans multiple sentences by trimming whitespace, converting to lowercase, collapsing extra spaces, and removing punctuation. |
| `03_tokenization.py` | Loads BERT and GPT-2 tokenizers, prints tokens and token IDs, decodes IDs, displays available BERT special tokens, and compares token counts. |
| `04_embeddings_similarity.py` | Generates sentence embeddings and compares a related sentence with an unrelated sentence using cosine similarity. |
| `05_semantic_search.py` | Embeds a small document collection, embeds three user queries, ranks documents by cosine similarity, and displays results. |

## 5. Important concepts

- **Training:** the model learns patterns from the training examples.
- **Inference:** the trained model uses what it learned to make a prediction on new text.
- **Preprocessing:** prepares text in a consistent format before later NLP steps.
- **Tokenization:** splits text into tokens. Subword tokenization can split a word into smaller pieces.
- **Special tokens:** tokens added by a tokenizer to mark roles or boundaries in a model input. BERT commonly uses `[CLS]` and `[SEP]`.
- **Embeddings:** numerical representations of text that capture aspects of meaning.
- **Cosine similarity:** compares the direction of two vectors; a higher score generally means the embeddings are more similar.
- **Word counts vs. embeddings:** word counts record term occurrence, while semantic embeddings represent text as numerical vectors that can capture meaning and context.

## 6. Evidence and submission

Run each program and capture screenshots of the code and its terminal output. Save your genuine screenshots in the `evidence/` folder. Use `evidence/README.md` to track what you have captured.

Do not submit empty or fabricated screenshots. Model downloads may make the first run slower.

## 7. Notes

The examples use small teaching datasets, so their evaluation scores are only demonstrations and should not be treated as real-world model performance.
