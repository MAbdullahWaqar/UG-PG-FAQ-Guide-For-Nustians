"""
Data Ingestion Module
=====================
Handles PDF parsing, section detection, and smart text chunking
with overlap for the UG and PG Handbooks.
"""

import os
import re
import pandas as pd
import pdfplumber


# ---------------------------------------------------------------------------
# Section header detection
# ---------------------------------------------------------------------------

# Patterns for section headers (e.g., "1.  Introduction", "3.2 Attendance")
SECTION_HEADER_PATTERNS = [
    re.compile(r"^(\d+\.?\d*\.?\d*)\s+([A-Z][A-Za-z\s,&\-/]+)"),   # "3.2 Attendance Policy"
    re.compile(r"^(Chapter|CHAPTER|Section|SECTION)\s+\d+", re.IGNORECASE),
    re.compile(r"^[A-Z][A-Z\s]{4,}$"),  # ALL CAPS lines (likely headers)
]


def _detect_section(line: str) -> str | None:
    """Return the section title if the line looks like a header, else None."""
    line = line.strip()
    if not line or len(line) < 3:
        return None
    for pat in SECTION_HEADER_PATTERNS:
        m = pat.match(line)
        if m:
            return line
    return None


def _is_section_boundary(line: str) -> bool:
    """Check if a line is a section header (used for hard chunk breaks)."""
    return _detect_section(line) is not None


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean extracted PDF text."""
    if not text:
        return ""
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    # Remove page-number-only lines
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_pages(pdf_path: str) -> list[dict]:
    """
    Extract text from each page of a PDF.
    Returns list of {'page': int, 'text': str}.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            # Also try to extract tables as text
            tables = page.extract_tables()
            table_text = ""
            if tables:
                for table in tables:
                    for row in table:
                        if row:
                            cells = [str(c) if c else "" for c in row]
                            table_text += " | ".join(cells) + "\n"
            combined = text
            if table_text:
                combined += "\n\n[Table Data]\n" + table_text
            pages.append({
                "page": i + 1,
                "text": clean_text(combined),
            })
    return pages


# ---------------------------------------------------------------------------
# Smart chunking — smaller, section-aware chunks
# ---------------------------------------------------------------------------

def chunk_pages(
    pages: list[dict],
    source: str,
    min_words: int = 80,
    max_words: int = 250,
    overlap_words: int = 30,
) -> list[dict]:
    """
    Split page texts into overlapping chunks of 80–250 words.
    
    KEY DESIGN: 
    - Smaller chunks (80-250 words) so each chunk is about ONE topic
    - Hard breaks at section headers so topics don't mix
    - Overlap for context continuity
    - Preserves page numbers and section headers
    """
    chunks = []
    chunk_id = 0

    # Build a list of (word, page_num, line_text, is_section_start)
    tokens_with_meta = []
    for pg in pages:
        lines = pg["text"].split("\n")
        for line in lines:
            is_section = _is_section_boundary(line)
            words = line.split()
            for w_idx, w in enumerate(words):
                tokens_with_meta.append((w, pg["page"], line, is_section and w_idx == 0))

    idx = 0
    current_section = "General"

    while idx < len(tokens_with_meta):
        # Check if we're at a section boundary
        if tokens_with_meta[idx][3]:
            current_section = tokens_with_meta[idx][2].strip()

        # Determine chunk end
        end_idx = min(idx + max_words, len(tokens_with_meta))

        # Hard break at section boundaries (don't let chunks cross sections)
        for j in range(idx + min_words, end_idx):
            if j < len(tokens_with_meta) and tokens_with_meta[j][3]:
                end_idx = j
                break

        # Try to find a sentence boundary near the end for cleaner splits
        if end_idx < len(tokens_with_meta) and end_idx > idx + min_words:
            for j in range(end_idx, max(idx + min_words, end_idx - 60), -1):
                if j < len(tokens_with_meta) and tokens_with_meta[j - 1][0].endswith((".", "?", "!", ":")):
                    end_idx = j
                    break

        chunk_words = tokens_with_meta[idx:end_idx]
        if not chunk_words:
            break

        # Extract metadata
        page_start = chunk_words[0][1]
        page_end = chunk_words[-1][1]
        text = " ".join(w for w, _, _, _ in chunk_words)

        # Detect section in this chunk
        for _, _, line, is_sec in chunk_words:
            if is_sec:
                sec = _detect_section(line)
                if sec:
                    current_section = sec

        chunk_id_str = f"{source}_p{page_start}_{chunk_id}"
        chunks.append({
            "chunk_id": chunk_id_str,
            "source": source,
            "page_start": page_start,
            "page_end": page_end,
            "section": current_section,
            "text": text,
            "word_count": len(chunk_words),
        })
        chunk_id += 1

        # Advance with overlap
        advance = max(end_idx - idx - overlap_words, min_words // 2)
        idx += advance

    return chunks


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------

def ingest_handbooks(handbook_dir: str = "Handbooks") -> pd.DataFrame:
    """
    Ingest all PDF handbooks and return a DataFrame of chunks.
    """
    all_chunks = []
    pdf_files = {
        "ug": os.path.join(handbook_dir, "ug.pdf"),
        "pg": os.path.join(handbook_dir, "pg.pdf"),
    }

    for source, path in pdf_files.items():
        if not os.path.exists(path):
            print(f"⚠️  Warning: {path} not found, skipping.")
            continue
        print(f"📄 Parsing {path}...")
        pages = extract_pages(path)
        print(f"   → Extracted {len(pages)} pages")
        chunks = chunk_pages(pages, source=source)
        print(f"   → Created {len(chunks)} chunks")
        all_chunks.extend(chunks)

    df = pd.DataFrame(all_chunks)
    print(f"\n✅ Total chunks: {len(df)}")
    return df


if __name__ == "__main__":
    df = ingest_handbooks()
    print(df.head())
    print(f"\nWord count stats:\n{df['word_count'].describe()}")
