import hashlib
import pickle
from collections import Counter

class SimHashRetriever:
    def __init__(self, hashbits=64):
        self.hashbits = hashbits
        self.chunks_data = []
        
    def _string_hash(self, v):
        """A stable string hash using md5."""
        if v == "":
            return 0
        x = int(hashlib.md5(v.encode('utf-8')).hexdigest(), 16)
        return x
        
    def get_features(self, text):
        """Extract word frequencies as features."""
        words = text.split()
        return Counter(words)
        
    def compute_simhash(self, text):
        """Compute the SimHash fingerprint for the text."""
        features = self.get_features(text)
        v = [0] * self.hashbits
        
        for feature, weight in features.items():
            h = self._string_hash(feature)
            for i in range(self.hashbits):
                bitmask = 1 << i
                if h & bitmask:
                    v[i] += weight
                else:
                    v[i] -= weight
                    
        fingerprint = 0
        for i in range(self.hashbits):
            if v[i] >= 0:
                fingerprint += 1 << i
                
        return fingerprint
        
    def hamming_distance(self, hash1, hash2):
        """Calculate the Hamming distance between two hashes."""
        x = (hash1 ^ hash2) & ((1 << self.hashbits) - 1)
        tot = 0
        while x:
            tot += 1
            x &= x - 1
        return tot
        
    def fit(self, df, text_column='processed_text'):
        """Compute and store SimHash for all chunks."""
        self.chunks_data = df.to_dict('records')
        
        for i, row in enumerate(self.chunks_data):
            text = row[text_column]
            self.chunks_data[i]['simhash'] = self.compute_simhash(text)
            
    def retrieve(self, query_processed, top_k=5):
        """Retrieve top_k chunks by comparing Hamming distance."""
        if not self.chunks_data:
            return []
            
        query_hash = self.compute_simhash(query_processed)
        
        results = []
        for chunk in self.chunks_data:
            dist = self.hamming_distance(query_hash, chunk['simhash'])
            
            result = chunk.copy()
            # Convert distance to similarity score (0 to 1)
            # Max distance is self.hashbits
            similarity = 1.0 - (dist / self.hashbits)
            
            result['hamming_distance'] = dist
            result['similarity_score'] = similarity
            result['retrieval_method'] = 'SimHash'
            
            results.append(result)
            
        # Sort by similarity (highest first) or distance (lowest first)
        results.sort(key=lambda x: x['hamming_distance'])
        
        # Remove the 'simhash' key from results before returning
        final_results = []
        for r in results[:top_k]:
            r.pop('simhash', None)
            final_results.append(r)
            
        return final_results
        
    def save(self, filepath="simhash_model.pkl"):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'hashbits': self.hashbits,
                'chunks_data': self.chunks_data
            }, f)
            
    def load(self, filepath="simhash_model.pkl"):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.hashbits = data['hashbits']
            self.chunks_data = data['chunks_data']
