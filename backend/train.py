"""
train.py
--------
This is where we CREATE OUR OWN MODEL from scratch.

Steps:
  1. Load the word/label data from training_words.csv
  2. Turn every word into features (using features.py)
  3. Split into a training set and a test set
  4. Train a Logistic Regression classifier on the training set
  5. Measure how well it does on the test set (data it never saw)
  6. Save the trained model to model/model.joblib so the web app can use it

We wrap the model in a Pipeline with a StandardScaler. The scaler puts every
feature on a comparable scale (e.g. "length" goes up to ~15 but "vowel_ratio"
is between 0 and 1). Logistic regression works much better when features are
scaled, and saving them together means the app does the exact same scaling.
"""

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix

from features import extract_features, FEATURE_NAMES

HERE = os.path.dirname(__file__)
DATA_CSV = os.path.join(HERE, "data", "training_words.csv")
MODEL_OUT = os.path.join(HERE, "model", "model.joblib")


def main():
    # 1. Load the data ------------------------------------------------------
    df = pd.read_csv(DATA_CSV)
    print(f"Loaded {len(df)} labeled words.")

    # 2. Build the feature matrix X and the label vector y ------------------
    #    X = list of feature vectors, one per word
    #    y = list of 0/1 labels
    X = [extract_features(str(w)) for w in df["word"]]
    y = df["label"].tolist()

    # 3. Train/test split ---------------------------------------------------
    #    We hold back 20% of the data to test on. stratify keeps the same
    #    ratio of complex/simple words in both halves. random_state makes the
    #    split reproducible so results don't change every run.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Build and train the model -----------------------------------------
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(X_train, y_train)

    # 5. Evaluate on the held-out test set ---------------------------------
    y_pred = model.predict(X_test)
    print("\n=== How well did the model do on unseen words? ===")
    print(classification_report(
        y_test, y_pred, target_names=["simple (0)", "complex (1)"]
    ))
    print("Confusion matrix [rows=actual, cols=predicted]:")
    print(confusion_matrix(y_test, y_pred))

    # Show which features mattered most (the learned weights).
    clf = model.named_steps["logisticregression"]
    print("\n=== What the model learned (feature weights) ===")
    for name, weight in zip(FEATURE_NAMES, clf.coef_[0]):
        direction = "harder" if weight > 0 else "easier"
        print(f"  {name:>14}: {weight:+.3f}  (higher value -> {direction})")

    # 6. Save the trained model --------------------------------------------
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"\nSaved trained model to {MODEL_OUT}")


if __name__ == "__main__":
    main()
