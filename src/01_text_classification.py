"""Week 03 - Phase 01: Machine Learning text classification."""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


def build_dataset():
    """Return a small example dataset and its sentiment labels."""
    texts = [
        "I love this course",
        "This course is excellent",
        "The lessons are very helpful",
        "I really enjoyed this class",
        "The teacher explains things clearly",
        "I hate this course",
        "This course is terrible",
        "The lessons are confusing",
        "I regret joining this class",
        "The teaching is very poor",
    ]
    labels = [
        "positive", "positive", "positive", "positive", "positive",
        "negative", "negative", "negative", "negative", "negative",
    ]
    return texts, labels


def train_and_evaluate():
    texts, labels = build_dataset()

    # Split the examples so the model is tested on unseen text.
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.30,
        random_state=42,
        stratify=labels,
    )

    vectorizer = CountVectorizer()
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_features, y_train)

    predictions = model.predict(X_test_features)

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    new_text = ["I really love these helpful lessons"]
    new_features = vectorizer.transform(new_text)
    new_prediction = model.predict(new_features)
    print("New text:", new_text[0])
    print("Prediction:", new_prediction[0])

    return model, vectorizer


if __name__ == "__main__":
    train_and_evaluate()
