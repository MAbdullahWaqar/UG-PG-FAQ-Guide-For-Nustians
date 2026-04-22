import os
import re
import pandas as pd
import pdfplumber

def clean_text(text):
    """Clean the extracted text by removing extra spaces and newlines."""
    if not text:
        return ""
    # Replace multiple whitespaces and newlines with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_and_chunk_pdf(pdf_path, doc_name, chunk_size=300):
    """
    Extract text from a PDF, clean it, and split into chunks of `chunk_size` words.
    Preserves document name, page number, and assigns a chunk ID.
    """
    chunks = []
    chunk_id_counter = 0
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if not text:
                    continue
                    
                text = clean_text(text)
                words = text.split()
                
                # Simple chunking by word count
                for i in range(0, len(words), chunk_size):
                    chunk_words = words[i:i + chunk_size]
                    chunk_text = " ".join(chunk_words)
                    
                    chunks.append({
                        "chunk_id": f"{doc_name}_p{page_num}_{chunk_id_counter}",
                        "document": doc_name,
                        "page_number": page_num,
                        "text": chunk_text
                    })
                    chunk_id_counter += 1
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
                
    return chunks

def ingest_handbooks(data_dir="Handbooks", chunk_size=300):
    """
    Ingest all PDFs in the given directory and return a DataFrame of chunks.
    """
    all_chunks = []
    
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found.")
        return pd.DataFrame()
        
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_dir, pdf_file)
        doc_name = os.path.splitext(pdf_file)[0]
        print(f"Ingesting {pdf_file}...")
        
        doc_chunks = extract_and_chunk_pdf(pdf_path, doc_name, chunk_size)
        all_chunks.extend(doc_chunks)
        
    df = pd.DataFrame(all_chunks)
    return df

if __name__ == "__main__":
    df = ingest_handbooks(chunk_size=300)
    print(f"Total chunks extracted: {len(df)}")
    # Save to a convenient format for later use
    df.to_csv("processed_chunks.csv", index=False)
    print("Chunks saved to processed_chunks.csv")
