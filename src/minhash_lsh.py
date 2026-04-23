from datasketch import MinHash, MinHashLSH
import pickle

class MinHashLSHRetriever:
    def __init__(self, num_perm=128, threshold=0.01): # Lowered threshold drastically for short query asymmetric search
        self.num_perm = num_perm
        self.threshold = threshold
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self.chunks_data = {}
        
    def get_shingles(self, text, k=1):
        """Convert text into k-word shingles. Using k=1 (unigrams) is better for asymmetric short queries."""
        words = text.split()
        if len(words) < k:
            return set([text])
        shingles = set()
        for i in range(len(words) - k + 1):
            shingle = " ".join(words[i:i+k])
            shingles.add(shingle)
        return shingles
        
    def compute_minhash(self, text):
        """Compute MinHash signature for text."""
        m = MinHash(num_perm=self.num_perm)
        shingles = self.get_shingles(text)
        for s in shingles:
            m.update(s.encode('utf8'))
        return m
        
    def fit(self, df, text_column='processed_text', id_column='chunk_id'):
        """Build the LSH index."""
        # Re-initialize LSH in case of re-fitting
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self.chunks_data = {}
        
        for idx, row in df.iterrows():
            chunk_id = row[id_column]
            text = row[text_column]
            
            # Store metadata
            self.chunks_data[chunk_id] = row.to_dict()
            
            # Compute MinHash and add to LSH
            m = self.compute_minhash(text)
            self.lsh.insert(chunk_id, m)
            
    def retrieve(self, query_processed, top_k=5):
        """Retrieve top_k chunks for a given query."""
        query_minhash = self.compute_minhash(query_processed)
        
        # Query LSH to get candidates
        candidates = self.lsh.query(query_minhash)
        
        if not candidates:
            return []
            
        # Rank candidates by estimated Jaccard similarity
        results = []
        for chunk_id in candidates:
            candidate_text = self.chunks_data[chunk_id]['processed_text']
            candidate_minhash = self.compute_minhash(candidate_text)
            
            # Estimate Jaccard similarity using MinHash
            similarity = query_minhash.jaccard(candidate_minhash)
            
            result = self.chunks_data[chunk_id].copy()
            result['similarity_score'] = similarity
            result['retrieval_method'] = 'MinHash_LSH'
            results.append(result)
            
        # Sort by similarity
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:top_k]
        
    def save(self, filepath="minhash_lsh_model.pkl"):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'num_perm': self.num_perm,
                'threshold': self.threshold,
                'lsh': self.lsh,
                'chunks_data': self.chunks_data
            }, f)
            
    def load(self, filepath="minhash_lsh_model.pkl"):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.num_perm = data['num_perm']
            self.threshold = data['threshold']
            self.lsh = data['lsh']
            self.chunks_data = data['chunks_data']
