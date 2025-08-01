# rag_system.py
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load the data we saved in the previous step
df = pd.read_csv('medquad_complete.csv')
knowledge_base = df['Answer'].tolist()

# Load a pre-trained embedding model
# The 'all-MiniLM-L6-v2' model is a great, lightweight choice for MacBooks
print("Loading Sentence-Transformer model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the knowledge base
print("Generating embeddings for the knowledge base...")
embeddings = embedding_model.encode(knowledge_base)

# Build the FAISS index for fast similarity search
print("Building FAISS index...")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

def retrieve_relevant_context(query, top_k=3):
    """
    Takes a user query, embeds it, and finds the top_k most
    relevant contexts from the knowledge base using FAISS.
    """
    query_embedding = embedding_model.encode([query])
    D, I = index.search(query_embedding, top_k)
    retrieved_contexts = [knowledge_base[i] for i in I[0]]
    return "\n\n".join(retrieved_contexts)

print("RAG system initialized and ready.") 