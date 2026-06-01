"""
prepare_data.py
---------------
Goal: produce ONE clean file -> data/training_words.csv  with two columns:

    word,label
    the,0
    thrombosis,1
    ...

where label = 1 means "complex / a patient probably won't understand it"
and   label = 0 means "simple / everyday word."

There are TWO ways this script can get its data:

  (A) REAL DATA (recommended for your final submission):
      Download the Complex Word Identification (CWI) 2018 English data and
      save a file as  data/cwi_raw.tsv.  This script will parse it.
      ---> CITE this dataset in your presentation. <---

  (B) BOOTSTRAP (works instantly, no download needed):
      If cwi_raw.tsv is NOT found, we build a starter dataset automatically:
        - the most COMMON English words  -> labeled 0 (simple)
        - the bundled medical_terms.txt  -> labeled 1 (complex)
      This lets you run the whole pipeline today, then swap in the real
      data later.

In BOTH cases we also merge in data/medical_terms.txt so the model always
sees medical vocabulary (that's our domain).
"""

import csv
import os
from wordfreq import top_n_list

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")
CWI_RAW = os.path.join(DATA, "cwi_raw.tsv")
MEDICAL_TERMS = os.path.join(DATA, "medical_terms.txt")
OUTPUT = os.path.join(DATA, "training_words.csv")


def load_medical_terms():
    """Read the bundled list of medical/hard words. All get label 1."""
    with open(MEDICAL_TERMS, encoding="utf-8") as f:
        words = [line.strip().lower() for line in f if line.strip()]
    return {w: 1 for w in words}


def parse_cwi(path):
    """
    Parse the CWI 2018 English data.

    `path` can be either a single .tsv file OR a folder that contains
    several .tsv files (which is how the official shared task data ships).

    CWI 2018 columns:
        0: HIT ID                          5: # native annotators
        1: sentence                        6: # non-native annotators
        2: start offset                    7: # natives marking complex
        3: end offset                      8: # non-natives marking complex
        4: target word/phrase              9: BINARY LABEL (0 or 1)  <-- we use this
                                          10: probabilistic label
    """
    # Step 1: figure out which file(s) to read.
    if os.path.isdir(path):
        files = []
        for root, _, names in os.walk(path):
            for fn in names:
                lower = fn.lower()
                if not lower.endswith(".tsv"):
                    continue
                # Skip test files (labels often missing) and non-English data.
                if "test" in lower:
                    continue
                if any(lang in lower for lang in ("spanish", "german", "french")):
                    continue
                # Keep the English source files.
                if any(src in lower for src in ("news", "wikinews", "wikipedia")):
                    files.append(os.path.join(root, fn))
        if not files:
            raise FileNotFoundError(
                f"No English training/dev .tsv files found inside {path}. "
                "Look for files named like News_Train.tsv, WikiNews_Train.tsv, "
                "Wikipedia_Train.tsv (and the _Dev.tsv versions)."
            )
        print(f"  Reading {len(files)} English CWI file(s):")
        for fn in files:
            print(f"    - {os.path.basename(fn)}")
    else:
        files = [path]

    # Step 2: parse each one, keeping only single alphabetic words.
    pairs = {}
    for fpath in files:
        with open(fpath, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 11:
                    continue
                target = row[4].strip().lower()
                if not target or " " in target or not target.isalpha():
                    continue
                try:
                    label = int(row[9])      # binary label column
                except (ValueError, IndexError):
                    continue
                pairs[target] = label
    return pairs

def bootstrap(num_common=600):
    """
    Build a starter dataset with no downloads:
      - top `num_common` everyday English words -> label 0
      - medical_terms.txt                       -> label 1
    """
    print("  cwi_raw.tsv not found -> using BOOTSTRAP dataset.")
    common = top_n_list("en", num_common)
    # Keep clean alphabetic words, drop anything too short to be interesting.
    data = {w.lower(): 0 for w in common if w.isalpha() and len(w) > 1}
    return data


def main():
    # Start from whichever base dataset is available.
    if os.path.exists(CWI_RAW):
        print("  Found cwi_raw.tsv -> parsing real CWI data.")
        data = parse_cwi(CWI_RAW)
    else:
        data = bootstrap()

    # Always merge in medical terms (label 1 wins for these).
    data.update(load_medical_terms())

    # Write the final CSV.
    os.makedirs(DATA, exist_ok=True)
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "label"])
        for word, label in sorted(data.items()):
            writer.writerow([word, label])

    n_complex = sum(1 for v in data.values() if v == 1)
    n_simple = len(data) - n_complex
    print(f"  Wrote {len(data)} words to training_words.csv "
          f"({n_complex} complex, {n_simple} simple).")


if __name__ == "__main__":
    main()
