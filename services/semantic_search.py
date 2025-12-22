import os
from turtle import Pen
from typing import Dict, List, Tuple

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from services.processing import load_movies

load_dotenv()

cache_dir = os.environ["CACHE_DIR"]

embedding_store_dir = f"{cache_dir}/movie_embeddings.npy"


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = dict()

    def search(self, query: str, limit: int):
        if self.embeddings is None or self.documents is None:
            raise ValueError(
                "No embeddings loaded. Call 'load_or_create_embeddings' first."
            )

        query_embedding = self.generate_embedding(query)

        dot_products = np.dot(self.embeddings, query_embedding)
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(
            query_embedding
        )
        cos_similarities = dot_products / norms
        top_indices = np.argsort(cos_similarities)[::-1][:limit]

        sorted_values = cos_similarities[top_indices]
        top_results = []
        for idx, score in zip(top_indices, sorted_values):
            doc = self.documents[idx]
            top_results.append(
                {
                    "score": float(score),
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )
        return top_results

    def generate_embedding(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            raise ValueError("text cannot be whitespace or empty")

        embeddings = self.model.encode([text])
        return embeddings[0]

    def load_or_create_embeddings(self, documents: List[Dict]):
        self.documents = documents

        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(embedding_store_dir):
            with open(embedding_store_dir, "rb") as f:
                self.embeddings = np.load(f)

            if len(self.embeddings) == len(self.documents):
                return self.embeddings
            else:
                return self.build_embeddings(documents)
        else:
            return self.build_embeddings(documents)

    def build_embeddings(self, documents: List[Dict]):
        self.documents = documents
        strs_to_embed = []
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            str_to_embed = f"{doc['title']}:{doc['description']}"
            strs_to_embed.append(str_to_embed)

        print("Generating embeddings...")
        embeddings = self.model.encode(strs_to_embed, show_progress_bar=True)
        self.embeddings = embeddings

        print("Saving embeddings...")
        with open(embedding_store_dir, "wb") as f:
            np.save(f, self.embeddings)

        print(f"Embeddings stored at {embedding_store_dir}")
        return self.embeddings


def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def verify_embeddings():
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)

    print(f"Number of docs: {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_text(text: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def cosine_similarity(vec1, vec2):
    dot_prd = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_prd / (norm1 * norm2)


def embed_query_text(query: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")
