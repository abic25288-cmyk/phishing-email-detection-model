from pathlib import Path
import sys
import joblib

MODEL = Path(__file__).resolve().parents[1] / "model.joblib"

def main():
    if not MODEL.exists():
        raise SystemExit("Model not found. Run: python -m src.train")
    text = " ".join(sys.argv[1:]).strip()
    if not text:
        raise SystemExit('Usage: python -m src.predict "email text"')
    model = joblib.load(MODEL)
    print("Prediction:", model.predict([text])[0])
    if hasattr(model, "predict_proba"):
        print("Confidence:", round(float(model.predict_proba([text]).max()), 3))

if __name__ == "__main__":
    main()
