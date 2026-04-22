import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

class ExtractiveAnswerGenerator:
    def __init__(self):
        pass
        
    def generate(self, query, top_chunks):
        """
        Generate an extractive answer by finding the most relevant sentences 
        from the top retrieved chunks.
        """
        if not top_chunks:
            return "No relevant information found."
            
        # Extract all sentences from the top-k chunks
        all_sentences = []
        for chunk in top_chunks:
            sentences = sent_tokenize(chunk['text'])
            all_sentences.extend(sentences)
            
        if not all_sentences:
            return "No readable sentences found in retrieved context."
            
        # Use TF-IDF to find the most similar sentence to the query
        vectorizer = TfidfVectorizer(stop_words='english')
        
        # Add query as the first item
        corpus = [query] + all_sentences
        
        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
            query_vec = tfidf_matrix[0:1]
            sentence_vecs = tfidf_matrix[1:]
            
            similarities = cosine_similarity(query_vec, sentence_vecs).flatten()
            
            # Get the top 3 sentences (to provide a bit more context)
            top_indices = similarities.argsort()[-3:][::-1]
            
            # Sort the indices so the sentences appear in their original order 
            # to maintain some logical flow
            top_indices_sorted = sorted(top_indices)
            
            best_sentences = [all_sentences[i] for i in top_indices_sorted if similarities[i] > 0.0]
            
            if not best_sentences:
                # Fallback: return the first few sentences of the best chunk
                return " ".join(sent_tokenize(top_chunks[0]['text'])[:3])
                
            return " ".join(best_sentences)
            
        except Exception as e:
            print(f"Error in answer generation: {e}")
            # Fallback
            return top_chunks[0]['text'][:300] + "..."
