"""
Text Preprocessing Module
=========================
Handles tokenization, lemmatization, stopword removal,
synonym expansion, and shingle generation for indexing.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)


# ---------------------------------------------------------------------------
# Synonym Expansion Map
# ---------------------------------------------------------------------------

SYNONYM_MAP = {
    "gpa":        ["gpa", "cgpa", "grade point average", "cumulative gpa"],
    "cgpa":       ["gpa", "cgpa", "grade point average", "cumulative gpa"],
    "fail":       ["fail", "failure", "failing", "failed"],
    "failure":    ["fail", "failure", "failing", "failed"],
    "repeat":     ["repeat", "retake", "re-enroll", "repeated"],
    "retake":     ["repeat", "retake", "re-enroll"],
    "attendance": ["attendance", "absent", "absence", "present", "presence"],
    "absent":     ["attendance", "absent", "absence"],
    "drop":       ["drop", "withdraw", "withdrawal", "dropped"],
    "withdraw":   ["drop", "withdraw", "withdrawal"],
    "semester":   ["semester", "term", "session"],
    "thesis":     ["thesis", "dissertation", "research"],
    "fee":        ["fee", "fees", "tuition", "charges", "payment"],
    "scholarship":["scholarship", "financial aid", "merit"],
    "probation":  ["probation", "academic warning", "warning"],
    "expel":      ["expel", "expulsion", "rustication", "dismiss"],
    "credit":     ["credit", "credits", "credit hour", "credit hours", "ch"],
    "exam":       ["exam", "examination", "test", "midterm", "final"],
    "degree":     ["degree", "program", "programme", "bachelor", "master"],
    "transfer":   ["transfer", "migration", "shift"],
}


# ---------------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------------

_stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()

# Additional domain stopwords that add noise
_domain_stop = {"university", "nust", "pakistan", "islamabad", "page", "student", "shall"}


def preprocess_text(text: str, expand_synonyms: bool = True) -> str:
    """
    Preprocess text: lowercase, remove punctuation, tokenize,
    remove stopwords, lemmatize, and optionally expand synonyms.

    Returns a cleaned string suitable for TF-IDF / SimHash.
    """
    if not text:
        return ""

    text = text.lower()
    # Remove punctuation but keep hyphens within words
    text = re.sub(r"[^\w\s\-]", " ", text)
    # Remove numbers that are standalone (keep numbers in context like "3.0")
    text = re.sub(r"\b\d+\b", "", text)

    tokens = word_tokenize(text)

    # Expand synonyms
    if expand_synonyms:
        expanded = []
        for token in tokens:
            if token in SYNONYM_MAP:
                expanded.extend(SYNONYM_MAP[token])
            else:
                expanded.append(token)
        tokens = expanded

    # Remove stopwords and lemmatize
    clean_tokens = [
        _lemmatizer.lemmatize(w)
        for w in tokens
        if w not in _stop_words
        and w not in _domain_stop
        and len(w) > 1
        and w not in string.punctuation
    ]

    return " ".join(clean_tokens)


def tokenize_to_set(text: str) -> set[str]:
    """
    Tokenize preprocessed text into a set of unique tokens.
    Used for MinHash set representation.
    """
    return set(text.split())


def generate_shingles(text: str, k: int = 3) -> set[str]:
    """
    Generate k-word shingles from preprocessed text.
    Shingles capture local word ordering and dramatically improve
    MinHash accuracy for short queries vs long documents.

    Args:
        text: Preprocessed text string
        k: Shingle size (number of words per shingle)

    Returns:
        Set of k-word shingle strings
    """
    words = text.split()
    if len(words) < k:
        # For very short texts, use individual words + pairs
        shingles = set(words)
        if len(words) >= 2:
            for i in range(len(words) - 1):
                shingles.add(f"{words[i]}_{words[i+1]}")
        return shingles

    shingles = set()
    for i in range(len(words) - k + 1):
        shingle = "_".join(words[i:i + k])
        shingles.add(shingle)
    # Also add individual words and bigrams for better short-query matching
    shingles.update(words)
    for i in range(len(words) - 1):
        shingles.add(f"{words[i]}_{words[i+1]}")
    return shingles


def preprocess_dataframe(df, text_col: str = "text") -> None:
    """
    Add preprocessed columns to the chunks DataFrame (in-place).
    """
    print("🔧 Preprocessing text chunks...")
    df["processed_text"] = df[text_col].apply(preprocess_text)
    df["token_set"] = df["processed_text"].apply(tokenize_to_set)
    df["shingles"] = df["processed_text"].apply(lambda x: generate_shingles(x, k=3))
    print(f"   → Preprocessed {len(df)} chunks")
    print(f"   → Avg tokens per chunk: {df['token_set'].apply(len).mean():.0f}")
    print(f"   → Avg shingles per chunk: {df['shingles'].apply(len).mean():.0f}")
