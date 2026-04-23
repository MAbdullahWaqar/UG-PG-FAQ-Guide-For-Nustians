import pandas as pd
from src.retrieval import QA_Retriever
df = pd.read_csv("processed_chunks.csv")
retriever = QA_Retriever()
retriever.fit_all(df)
print("\n--- TF-IDF ---")
res = retriever.retrieve("What is Min GPA requirements", method='TF-IDF', top_k=5)
for r in res: print(f"{r['chunk_id']}: {r['similarity_score']:.4f}")
print("\n--- MinHash ---")
res2 = retriever.retrieve("What is Min GPA requirements", method='MinHash_LSH', top_k=5)
for r in res2: print(f"{r['chunk_id']}: {r['similarity_score']:.4f}")
print("\n--- SimHash ---")
res3 = retriever.retrieve("What is Min GPA requirements", method='SimHash', top_k=5)
for r in res3: print(f"{r['chunk_id']}: {r['similarity_score']:.4f}")
