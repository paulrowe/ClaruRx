"""
app.py
------
The web server (backend). It does three jobs:

  1. Loads our trained model (model/model.joblib) and the plain-language
     glossary (glossary.json) when the server starts.
  2. Exposes a POST endpoint /analyze that takes a chunk of medical text.
  3. For that text it: splits it into words, asks OUR MODEL which words are
     complex, attaches a plain-language definition when we have one, and
     calculates how hard the text is to read (Flesch-Kincaid).

Run it with:   uvicorn app:app --reload
Then it lives at http://127.0.0.1:8000  (interactive docs at /docs).
"""

import json
import os
import re

import joblib
import textstat
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from features import extract_features

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "model", "model.joblib")
GLOSSARY_PATH = os.path.join(HERE, "glossary.json")

# Words the model flags this confidently (or higher) are marked complex.
# Lower it to flag more words; raise it to flag fewer. Tune to taste.
COMPLEX_THRESHOLD = 0.5

# --- Load everything ONCE at startup (not on every request) ---------------
model = joblib.load(MODEL_PATH)
with open(GLOSSARY_PATH, encoding="utf-8") as f:
    glossary = json.load(f)

app = FastAPI(title="ClariMed API")

# Allow the React dev server (and your deployed frontend) to call this API.
# In production, replace "*" with your actual frontend URL for security.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str


def is_word(token):
    """True if the token is an actual alphabetic word (not punctuation/space)."""
    return bool(re.match(r"[A-Za-z]+$", token))


@app.get("/")
def health_check():
    """A simple endpoint so you can confirm the server is alive."""
    return {"status": "ok", "message": "ClariMed API is running"}


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    text = request.text

    # Split into pieces but KEEP the punctuation and spaces, so the frontend
    # can rebuild the original text exactly. The parentheses in the regex make
    # re.split include the separators in the result.
    pieces = re.split(r"([^A-Za-z]+)", text)

    tokens = []
    complex_count = 0
    word_count = 0

    for piece in pieces:
        if not piece:
            continue

        if is_word(piece):
            word_count += 1
            # Ask OUR model: how likely is this word to be complex?
            features = [extract_features(piece)]
            probability = float(model.predict_proba(features)[0][1])
            is_complex = probability >= COMPLEX_THRESHOLD

            definition = glossary.get(piece.lower())
            if is_complex:
                complex_count += 1

            tokens.append({
                "text": piece,
                "isWord": True,
                "complex": is_complex,
                "probability": round(probability, 3),
                "definition": definition,
            })
        else:
            # Punctuation / spaces: pass through untouched.
            tokens.append({"text": piece, "isWord": False, "complex": False})

    # Readability of the WHOLE text (Flesch-Kincaid grade + reading ease).
    readability = {
        "gradeLevel": round(textstat.flesch_kincaid_grade(text), 1),
        "readingEase": round(textstat.flesch_reading_ease(text), 1),
    }

    return {
        "tokens": tokens,
        "stats": {
            "totalWords": word_count,
            "complexWords": complex_count,
            "percentComplex": round(100 * complex_count / word_count, 1) if word_count else 0,
        },
        "readability": readability,
    }
