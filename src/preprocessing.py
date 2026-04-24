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
# Typo Correction Dictionary
# ---------------------------------------------------------------------------
# Common misspellings in student queries about academic policies.

TYPO_CORRECTIONS = {
    # Attendance
    "attendence": "attendance", "attendace": "attendance", "attendane": "attendance",
    "attandance": "attendance", "attendnce": "attendance", "atendance": "attendance",
    # Semester
    "semster": "semester", "semeseter": "semester", "semster": "semester",
    "semestr": "semester", "semestre": "semester", "semmester": "semester",
    # Scholarship
    "scholorship": "scholarship", "scholaship": "scholarship", "scolarship": "scholarship",
    "scholarshp": "scholarship", "scholership": "scholarship",
    # Examination
    "examinaton": "examination", "examiantion": "examination", "examnation": "examination",
    "examinaion": "examination", "examiation": "examination",
    # Registration
    "registeration": "registration", "registation": "registration",
    "regestration": "registration", "registraion": "registration",
    # Graduation
    "graduaton": "graduation", "graduaion": "graduation", "gradution": "graduation",
    # Probation
    "probaton": "probation", "probaion": "probation", "probtion": "probation",
    # Requirements
    "requirment": "requirement", "requirments": "requirements",
    "requiremnt": "requirement", "reqirement": "requirement",
    # Courses
    "corse": "course", "coarse": "course", "cours": "course", "cources": "courses",
    # Transfer
    "tranfer": "transfer", "trasfer": "transfer", "transfr": "transfer",
    # Department
    "departmnt": "department", "deparment": "department", "departmet": "department",
    # Thesis / Dissertation
    "theses": "thesis", "theisis": "thesis", "disertation": "dissertation",
    "disseration": "dissertation", "dissertaton": "dissertation",
    # Grade / Grading
    "gradeing": "grading", "gradng": "grading",
    # Minimum / Maximum
    "minimun": "minimum", "minium": "minimum", "minmum": "minimum",
    "maximun": "maximum", "maxium": "maximum",
    # GPA / CGPA
    "cg pa": "cgpa", "c.g.p.a": "cgpa", "g.p.a": "gpa",
    # Miscellaneous
    "refud": "refund", "refnd": "refund", "rефund": "refund",
    "hostle": "hostel", "hostl": "hostel",
    "discpline": "discipline", "dicipline": "discipline",
    "convocaton": "convocation", "convoction": "convocation",
    "transcirpt": "transcript", "transcipt": "transcript",
    "credithour": "credit hour", "credithr": "credit hour",
    "rustication": "rustication", "rusticaton": "rustication",
    "deffered": "deferred", "defered": "deferred",
    "withdrawl": "withdrawal", "withdrawel": "withdrawal",
    "enrolment": "enrollment", "enrolement": "enrollment",
}


def normalize_query(query: str) -> str:
    """
    Normalize a user query by correcting common typos/misspellings.
    Applied BEFORE preprocessing to ensure the synonym map and
    TF-IDF vocabulary can match the corrected terms.
    """
    words = query.lower().split()
    corrected = []
    for w in words:
        corrected.append(TYPO_CORRECTIONS.get(w, w))
    return " ".join(corrected)


# ---------------------------------------------------------------------------
# Synonym Expansion Map (trimmed — only core equivalences)
# ---------------------------------------------------------------------------

SYNONYM_MAP = {
    "gpa":          ["gpa", "cgpa"],
    "cgpa":         ["gpa", "cgpa"],
    "fail":         ["fail", "failure", "failed"],
    "failure":      ["fail", "failure", "failed"],
    "failed":       ["fail", "failure", "failed"],
    "repeat":       ["repeat", "retake", "repeated", "repetition"],
    "retake":       ["repeat", "retake", "repeated", "repetition"],
    "repetition":   ["repeat", "retake", "repeated", "repetition"],
    "drop":         ["drop", "withdraw", "withdrawal"],
    "withdraw":     ["drop", "withdraw", "withdrawal"],
    "withdrawal":   ["drop", "withdraw", "withdrawal"],
    "fee":          ["fee", "fees", "tuition"],
    "fees":         ["fee", "fees", "tuition"],
    "tuition":      ["fee", "fees", "tuition"],
    "thesis":       ["thesis", "dissertation"],
    "dissertation": ["thesis", "dissertation"],
    "exam":         ["exam", "examination"],
    "examination":  ["exam", "examination"],
    "probation":    ["probation", "warning"],
    "expelled":     ["expelled", "rusticated", "dismissed"],
    "rusticated":   ["expelled", "rusticated", "dismissed"],
    "dismissed":    ["expelled", "rusticated", "dismissed"],
    "internship":   ["internship", "industrial training"],
    "marks":        ["marks", "grades", "percentage"],
    "passing":      ["passing", "minimum", "requirement"],
    "freeze":       ["freeze", "defer", "deferment"],
    "defer":        ["freeze", "defer", "deferment"],
    "deferment":    ["freeze", "defer", "deferment"],
    "hostel":       ["hostel", "accommodation", "residence"],
}

# NOTE: We do NOT expand "attendance", "semester", "credit", "degree", etc.
# because they are already common in academic documents and expansion
# just adds noise. We only expand where there's a genuine vocabulary gap.


# ---------------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------------

_stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()

# VERY minimal domain stopwords — be careful not to remove query-important words
_domain_stop = {"nust", "page", "handbook"}


def preprocess_text(text: str, expand_synonyms: bool = True) -> str:
    """
    Preprocess text: lowercase, clean, tokenize,
    remove stopwords, lemmatize, and optionally expand synonyms.

    CRITICAL: We KEEP numbers (2.0, 75, 3.50) because they encode
    policy thresholds (GPA requirements, attendance percentages, etc.)
    """
    if not text:
        return ""

    text = text.lower()
    # Remove punctuation but keep hyphens within words and decimal numbers
    text = re.sub(r"[^\w\s\-\.]", " ", text)
    # Clean up periods that are NOT part of decimal numbers (e.g., end of sentence)
    text = re.sub(r"\.(\s|$)", r" \1", text)
    # Clean up remaining odd whitespace
    text = re.sub(r"\s+", " ", text)

    tokens = word_tokenize(text)

    # Expand synonyms
    if expand_synonyms:
        expanded = []
        for token in tokens:
            if token in SYNONYM_MAP:
                # Add all synonyms (deduped)
                expanded.extend(SYNONYM_MAP[token])
            else:
                expanded.append(token)
        tokens = expanded

    # Remove stopwords and lemmatize
    # KEEP: numbers, domain terms like "student", "minimum", "attendance"
    clean_tokens = []
    for w in tokens:
        if w in _stop_words and w not in ("no", "not"):
            continue
        if w in _domain_stop:
            continue
        if len(w) < 2 and not w.isdigit():
            continue
        if w in string.punctuation:
            continue
        clean_tokens.append(_lemmatizer.lemmatize(w))

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
