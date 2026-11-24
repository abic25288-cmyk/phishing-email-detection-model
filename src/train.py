from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction import DictVectorizer
from src.features import extract_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "emails.csv"
MODEL = ROOT / "model.joblib"

def feature_dicts(values):
    return [extract_features(value) for value in values]

def main():
    df = pd.read_csv(DATA).dropna(subset=["text", "label"])
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )
    from sklearn.pipeline import FeatureUnion
    from sklearn.preprocessing import FunctionTransformer
    features = FeatureUnion([
        ("text", TfidfVectorizer(ngram_range=(1, 2), max_features=3000)),
        ("signals", Pipeline([
            ("extract", FunctionTransformer(feature_dicts, validate=False)),
            ("vectorize", DictVectorizer()),
        ])),
    ])
    model = Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("Confusion matrix [phishing, safe] order follows sorted labels:")
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions, zero_division=0))
    joblib.dump(model, MODEL)
    print(f"Saved model to {MODEL}")

if __name__ == "__main__":
    main()
