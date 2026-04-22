import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class TFIDFRetriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.chunks_data = None
        
    def fit(self, df, text_column='processed_text'):
        """Fit the TF-IDF model and transform the corpus."""
        self.chunks_data = df.to_dict('records')
        corpus = df[text_column].tolist()
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        
    def retrieve(self, query_processed, top_k=5):
        """Retrieve the top_k most similar chunks for a given query."""
        if self.tfidf_matrix is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        query_vec = self.vectorizer.transform([query_processed])
        
        # Compute cosine similarity between query and all chunks
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top-k indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            result = self.chunks_data[idx].copy()
            result['similarity_score'] = similarities[idx]
            result['retrieval_method'] = 'TF-IDF'
            results.append(result)
            
        return results

    def save(self, filepath="tfidf_model.pkl"):
        """Save the model and index."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'tfidf_matrix': self.tfidf_matrix,
                'chunks_data': self.chunks_data
            }, f)
            
    def load(self, filepath="tfidf_model.pkl"):
        """Load the model and index."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.tfidf_matrix = data['tfidf_matrix']
            self.chunks_data = data['chunks_data']
