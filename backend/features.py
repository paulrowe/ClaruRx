"""
features.py
-----------
This module turns a single WORD into a list of numbers (a "feature vector").

A logistic-regression model cannot read words. It can only do math on numbers.
So our job here is to describe each word with a handful of numbers that capture
"what makes a word hard for a patient to understand."

These features are the part of the project that is most clearly MY OWN work.
I chose each one for a reason, and I can explain all of them:

  1. length            -> longer words tend to be harder ("hypertension" vs "high")
  2. syllable_count    -> more syllables = harder to read out loud / decode
  3. rarity            -> how rare the word is in everyday English (rare = harder)
  4. has_medical_suffix-> medical jargon often ends in -itis, -osis, -emia, etc.
  5. vowel_ratio       -> a rough measure of how "pronounceable" a word is

If you want to improve the model later, adding or changing features here is the
single most effective thing you can do.
"""

from wordfreq import zipf_frequency

# Common endings that signal medical / technical jargon.
# A word ending in one of these is very likely to be a term a patient won't know.
MEDICAL_SUFFIXES = (
    "itis",    # inflammation        (e.g. arthritis, bronchitis)
    "osis",    # condition / disease (e.g. thrombosis, fibrosis)
    "emia",    # blood condition     (e.g. anemia, leukemia)
    "ectomy",  # surgical removal    (e.g. appendectomy)
    "otomy",   # surgical cut        (e.g. tracheotomy)
    "ostomy",  # surgical opening    (e.g. colostomy)
    "pathy",   # disease             (e.g. neuropathy)
    "plasty",  # surgical repair     (e.g. angioplasty)
    "scopy",   # examination         (e.g. endoscopy)
    "gram",    # recording / image   (e.g. mammogram)
    "algia",   # pain                (e.g. neuralgia)
)


def count_syllables(word):
    """
    Count syllables using a simple, readable rule:
    count each GROUP of vowels as one syllable, then subtract a silent
    trailing 'e'. This isn't perfect, but it's transparent and good enough,
    and I can explain exactly how it works.
    """
    word = word.lower().strip()
    if not word:
        return 0

    vowels = "aeiouy"
    count = 0
    previous_was_vowel = False

    for char in word:
        is_vowel = char in vowels
        # Only count the START of each vowel group (e.g. "ea" = 1 syllable).
        if is_vowel and not previous_was_vowel:
            count += 1
        previous_was_vowel = is_vowel

    # A trailing silent 'e' (like in "make") usually isn't its own syllable.
    if word.endswith("e") and count > 1:
        count -= 1

    # Every real word has at least one syllable.
    return max(count, 1)


def vowel_ratio(word):
    """Fraction of letters that are vowels. Very low/very high can signal odd words."""
    word = word.lower().strip()
    if not word:
        return 0.0
    vowels = sum(1 for c in word if c in "aeiou")
    return vowels / len(word)


def has_medical_suffix(word):
    """Return 1 if the word ends in a known medical suffix, else 0."""
    word = word.lower().strip()
    return 1 if word.endswith(MEDICAL_SUFFIXES) else 0


def word_rarity(word):
    """
    How rare is this word in everyday English?

    wordfreq's zipf_frequency returns a number roughly 0-8:
        ~7 = extremely common ("the", "is")
        ~4 = normal everyday word ("doctor", "happy")
        ~2 = rare / technical ("thrombosis")
        ~0 = essentially never seen

    We FLIP it so that a HIGHER number means HARDER, which is more intuitive:
        rarity = 8 - zipf_frequency
    """
    z = zipf_frequency(word.lower().strip(), "en")
    return 8.0 - z


# The fixed ORDER of features. Training and prediction must use the same order,
# so we define it once, here, and reuse it everywhere.
FEATURE_NAMES = ["length", "syllables", "rarity", "medical_suffix", "vowel_ratio"]


def extract_features(word):
    """Turn one word into a list of numbers, in FEATURE_NAMES order."""
    word = word.lower().strip()
    return [
        len(word),                 # length
        count_syllables(word),     # syllables
        word_rarity(word),         # rarity
        has_medical_suffix(word),  # medical_suffix
        vowel_ratio(word),         # vowel_ratio
    ]


# Quick manual test: run "python features.py" to sanity-check the numbers.
if __name__ == "__main__":
    for w in ["the", "doctor", "hypertension", "thrombosis", "cat"]:
        print(f"{w:>14} -> {dict(zip(FEATURE_NAMES, extract_features(w)))}")
