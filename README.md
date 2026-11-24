# Phishing Email Detection Model

A beginner-friendly machine-learning project that classifies email text as **safe** or **phishing** using TF-IDF text features and simple URL/keyword features.

## Features
- Text vectorization with TF-IDF
- Lightweight URL and suspicious-language features
- Logistic Regression classifier
- Train/test split with accuracy and confusion matrix
- Save and load a trained model
- Command-line prediction script

## Run locally
```bash
python -m venv .venv
# activate the environment
pip install -r requirements.txt
python -m src.train
python -m src.predict "Your email text goes here"
``

The included CSV is a small synthetic teaching dataset. It is not suitable for real-world filtering or security decisions. For meaningful evaluation, use a larger, carefully labelled dataset and check precision, recall, false positives, and false negatives.
