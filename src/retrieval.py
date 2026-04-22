from src.preprocessing import preprocess_text
from src.tfidf_baseline import TFIDFRetriever
from src.minhash_lsh import MinHashLSHRetriever
from src.simhash import SimHashRetriever

class QA_Retriever:
    def __init__(self):
        self.tfidf = TFIDFRetriever()
        self.minhash_lsh = MinHashLSHRetriever()
        self.simhash = SimHashRetriever()
        
    def fit_all(self, df):
        """Fit all retrieval models using the provided DataFrame."""
        print("Fitting TF-IDF model...")
        self.tfidf.fit(df)
        print("Fitting MinHash+LSH model...")
        self.minhash_lsh.fit(df)
        print("Fitting SimHash model...")
        self.simhash.fit(df)
        
    def retrieve(self, query, method='TF-IDF', top_k=5):
        """
        Process query and retrieve top_k chunks using the specified method.
        method: 'TF-IDF', 'MinHash_LSH', or 'SimHash'
        """
        processed_query = preprocess_text(query)
        
        if method == 'TF-IDF':
            return self.tfidf.retrieve(processed_query, top_k)
        elif method == 'MinHash_LSH':
            return self.minhash_lsh.retrieve(processed_query, top_k)
        elif method == 'SimHash':
            return self.simhash.retrieve(processed_query, top_k)
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
            
    def retrieve_with_reranking(self, query, method='TF-IDF', top_k=5, alpha=0.7):
        """
        Extension: Rerank results combining similarity score and position importance.
        We'll assume chunks appearing earlier in a page/document have slightly higher importance.
        """
        results = self.retrieve(query, method, top_k=top_k*2) # Get more candidates for reranking
        
        if not results:
            return []
            
        for idx, res in enumerate(results):
            # Calculate a position score (1.0 for first, decreasing for later chunks)
            # using chunk_id which contains the chunk number (e.g., ug_p1_0)
            try:
                chunk_num = int(res['chunk_id'].split('_')[-1])
                # Simple decay function based on chunk position
                position_score = 1.0 / (1.0 + 0.1 * chunk_num)
            except:
                position_score = 0.5
                
            # Combine scores
            res['original_score'] = res['similarity_score']
            res['rerank_score'] = (alpha * res['similarity_score']) + ((1 - alpha) * position_score)
            
        # Sort by new rerank_score
        results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return results[:top_k]
