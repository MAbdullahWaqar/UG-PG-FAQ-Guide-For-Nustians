import time
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.retrieval import QA_Retriever

class Evaluator:
    def __init__(self, df):
        self.df = df
        self.retriever = QA_Retriever()
        self.retriever.fit_all(df)
        
    def get_size_in_mb(self, obj):
        # A rough estimate for Python objects
        import pickle
        try:
            return len(pickle.dumps(obj)) / (1024 * 1024)
        except:
            return sys.getsizeof(obj) / (1024 * 1024)
        
    def evaluate_queries(self, queries, top_k=5):
        """
        Evaluate exact vs approx methods. 
        Uses TF-IDF top-K as ground truth.
        """
        results = []
        
        methods = ['TF-IDF', 'MinHash_LSH', 'SimHash']
        
        # Calculate index sizes
        index_sizes = {
            'TF-IDF': self.get_size_in_mb(self.retriever.tfidf.tfidf_matrix),
            'MinHash_LSH': self.get_size_in_mb(self.retriever.minhash_lsh.lsh),
            'SimHash': self.get_size_in_mb(self.retriever.simhash.chunks_data)
        }
        
        for query in queries:
            query_results = {}
            
            for method in methods:
                start_time = time.time()
                retrieved = self.retriever.retrieve(query, method=method, top_k=top_k)
                latency = time.time() - start_time
                
                chunk_ids = [res['chunk_id'] for res in retrieved]
                query_results[method] = {
                    'chunk_ids': chunk_ids,
                    'latency': latency
                }
                
            # Calculate Precision and Recall using TF-IDF as Ground Truth
            gt_set = set(query_results['TF-IDF']['chunk_ids'])
            
            for method in methods:
                if method == 'TF-IDF':
                    precision = 1.0
                    recall = 1.0
                else:
                    retrieved_set = set(query_results[method]['chunk_ids'])
                    intersection = gt_set.intersection(retrieved_set)
                    
                    precision = len(intersection) / len(retrieved_set) if retrieved_set else 0.0
                    recall = len(intersection) / len(gt_set) if gt_set else 0.0
                    
                results.append({
                    'Query': query,
                    'Method': method,
                    'Precision@k': precision,
                    'Recall@k': recall,
                    'Latency (s)': query_results[method]['latency'],
                    'Index Size (MB)': index_sizes[method]
                })
                
        return pd.DataFrame(results)
        
    def run_scalability_test(self, query, multipliers=[1, 2, 5]):
        """Duplicate corpus to test scalability."""
        results = []
        
        for mult in multipliers:
            print(f"Testing scalability with {mult}x corpus size...")
            # Duplicate dataframe
            df_scaled = pd.concat([self.df] * mult, ignore_index=True)
            # Need unique chunk ids
            df_scaled['chunk_id'] = [f"{row['chunk_id']}_{i}" for i, row in df_scaled.iterrows()]
            
            # Re-fit
            temp_retriever = QA_Retriever()
            temp_retriever.fit_all(df_scaled)
            
            methods = ['TF-IDF', 'MinHash_LSH', 'SimHash']
            for method in methods:
                start_time = time.time()
                temp_retriever.retrieve(query, method=method, top_k=5)
                latency = time.time() - start_time
                
                results.append({
                    'Corpus Multiplier': mult,
                    'Total Chunks': len(df_scaled),
                    'Method': method,
                    'Latency (s)': latency
                })
                
        return pd.DataFrame(results)
