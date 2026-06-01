# ClariMed

A plain-language medical reader. Paste a medical document, lab result, or
discharge note, and ClariMed highlights the words a patient is unlikely to
understand, gives them plain-language meanings, and scores how hard the text is
to read.

The "intelligence" is a **logistic-regression classifier I trained from
scratch** to detect complex medical vocabulary. Supporting tools (a glossary
and a readability formula) act on what the model flags.

## Tech stack
- **Model:** scikit-learn (Logistic Regression), trained from scratch
- **Backend:** FastAPI + Python
- **Frontend:** React + Vite
- **Helpers:** wordfreq (features), textstat (readability)

## Quick start
**Backend**
```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python prepare_data.py             # builds the training data
python train.py                    # trains + saves the model
uvicorn app:app --reload           # starts the API at http://127.0.0.1:8000
```
**Frontend** (in a second terminal)
```bash
cd frontend
npm install
npm run dev                        # opens http://localhost:5173
```

See **BUILD_GUIDE.md** for the full, step-by-step walkthrough.

## Disclaimer
ClariMed is a reading aid, not medical advice. It does not replace a doctor,
nurse, or pharmacist.

## Citations
- Complex Word Identification (CWI) Shared Task 2018 dataset — training labels.
- PLABA (Plain Language Adaptation of Biomedical Abstracts) — medical terms.
- Libraries: scikit-learn, FastAPI, React, Vite, wordfreq, textstat.
