import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd

# Download required NLTK resources silently
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

def preprocess_text(text):
    """
    Preprocess text: lowercase, remove punctuation and stopwords,
    tokenize, and lemmatize.
    """
    if not isinstance(text, str):
        return ""
        
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in stop_words and word.isalpha()
    ]
    
    return " ".join(clean_tokens)

def preprocess_chunks(df):
    """
    Apply preprocessing to the 'text' column of a DataFrame.
    Returns a new DataFrame with a 'processed_text' column.
    """
    df_processed = df.copy()
    df_processed['processed_text'] = df_processed['text'].apply(preprocess_text)
    return df_processed

if __name__ == "__main__":
    # Test the preprocessing if executed directly
    sample_text = "The quick brown foxes are jumping over the lazy dogs!"
    print(f"Original: {sample_text}")
    print(f"Processed: {preprocess_text(sample_text)}")
