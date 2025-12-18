import math
import os
from collections import Counter
from pickle import dump, load
from typing import Dict, List

from dotenv import load_dotenv

from services.processing import load_movies, tokenize

from .utils import BM25_B, BM25_K1

load_dotenv()

cache_dir = os.environ["CACHE_DIR"]
index_dir = f"{cache_dir}/index.pkl"
docmap_dir = f"{cache_dir}/docmap.pkl"
tf_dir = f"{cache_dir}/term_frequencies.pkl"
doc_lengths_dir = f"{cache_dir}/doc_lengths.pkl"


class InvertedIndex:
    def __init__(self):
        self.index: Dict[str, set] = {}
        self.docmap: Dict[int, Dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.docs_length: Dict[int, int] = {}

    def __add_document(self, doc_id: int, text: str):
        text_tks = tokenize(text)
        for tk in text_tks:
            self.index.setdefault(tk, set()).add(doc_id)

        self.term_frequencies[doc_id] = Counter(text_tks)
        self.docs_length[doc_id] = len(text_tks)

    def get_documents(self, term: str) -> List[int]:
        term = term.lower()
        doc_ids = self.index.get(term, set())
        return sorted(doc_ids)

    def search(self, tokens: List[str], max_results: int = 5) -> List[Dict]:
        matching_docs = []
        for token in tokens:
            doc_ids = self.get_documents(token)
            for id in doc_ids:
                matching_docs.append(self.docmap.get(id))
                if len(matching_docs) == max_results:
                    return matching_docs
        return matching_docs

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize(term)
        if len(tokens) > 1:
            raise ValueError("term must be a single word!")

        token = tokens[0]
        doc_counter = self.term_frequencies.get(doc_id, Counter())
        return doc_counter.get(token, 0)

    def get_idf(self, term: str) -> float:
        tokens = tokenize(term)
        if len(tokens) > 1:
            raise ValueError("term must be a single word!")

        token = tokens[0]
        n_docs = len(self.docmap)
        n_matching_docs = len(self.get_documents(token))
        idf = math.log((n_docs + 1) / (n_matching_docs + 1))
        return idf

    def bm25(self, doc_id: int, term: str) -> float:
        tokens = tokenize(term)
        if len(tokens) > 1:
            raise ValueError("term must be a single word!")

        token = tokens[0]
        bm25_tf = self.get_bm25_tf(doc_id, token)
        bm25_idf = self.get_bm25_idf(token)

        return bm25_tf * bm25_idf

    def bm25_search(self, query: str, limit: int) -> Dict[int, float]:
        tokens = tokenize(query)
        scores: Dict[int, float] = {}
        for token in tokens:
            doc_ids = self.get_documents(token)
            for doc_id in doc_ids:
                scores[doc_id] = scores.get(doc_id, 0.0) + self.bm25(doc_id, token)

        sorted_scores = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        return dict(sorted_scores[:limit])

    def get_bm25_tf(self, doc_id: int, term: str, k1=BM25_K1, b=BM25_B) -> float:
        raw_tf = self.get_tf(doc_id, term)

        doc_length = self.docs_length.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()

        length_norm = 1 - b + b * (doc_length / avg_doc_length)

        bm25_tf = (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)
        return bm25_tf

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize(term)
        if len(tokens) > 1:
            raise ValueError("term must be a single word!")

        n_docs = len(self.docmap)

        token = tokens[0]
        n_matching_docs = len(self.get_documents(token))

        bm25_idf = math.log(
            (n_docs - n_matching_docs + 0.5) / (n_matching_docs + 0.5) + 1
        )
        return bm25_idf

    def __get_avg_doc_length(self) -> float:
        if not self.docs_length:
            return 0.0

        total_doc_length = sum(self.docs_length.values())
        avg_doc_length = total_doc_length / len(self.docs_length)
        return avg_doc_length

    def build(self):
        movies = load_movies()
        for movie in movies:
            id = movie["id"]
            title = movie["title"]
            description = movie["description"]

            self.__add_document(id, text=f"{title} {description}")
            self.docmap[id] = movie

    def save(self):
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)

        with open(index_dir, "wb") as f:
            dump(self.index, f)

        with open(docmap_dir, "wb") as f:
            dump(self.docmap, f)

        with open(tf_dir, "wb") as f:
            dump(self.term_frequencies, f)

        with open(doc_lengths_dir, "wb") as f:
            dump(self.docs_length, f)

    def load(self):
        dirs = [index_dir, docmap_dir, tf_dir, doc_lengths_dir]
        for dir in dirs:
            if not os.path.exists(dir):
                raise FileNotFoundError(f"Could not find {dir}")

        with open(index_dir, "rb") as f:
            self.index = load(f)

        with open(docmap_dir, "rb") as f:
            self.docmap = load(f)

        with open(tf_dir, "rb") as f:
            self.term_frequencies = load(f)

        with open(doc_lengths_dir, "rb") as f:
            self.docs_length = load(f)
