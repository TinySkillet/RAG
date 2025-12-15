from typing import Dict, List
from services.processing import tokenize, load_movies
from pickle import dump, load
from dotenv import load_dotenv
from collections import Counter
import os
import math

load_dotenv()

cache_dir = os.environ["CACHE_DIR"]
index_dir = f"{cache_dir}/index.pkl"
docmap_dir = f"{cache_dir}/docmap.pkl"
tf_dir = f"{cache_dir}/term_frequencies.pkl"


class InvertedIndex:
    def __init__(self):
        self.index: Dict[str, set] = {}
        self.docmap: Dict[int, Dict] = {}
        self.term_frequencies: dict[int, Counter] = {}

    def __add_document(self, doc_id: int, text: str):
        text_tks = tokenize(text)
        for tk in text_tks:
            self.index.setdefault(tk, set()).add(doc_id)

        self.term_frequencies[doc_id] = Counter(text_tks)

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

    def get_idf(self, term: str) -> int:
        tokens = tokenize(term)
        if len(tokens) > 1:
            raise ValueError("term must be a single word!")

        token = tokens[0]
        n_matching_docs = len(self.get_documents(token))
        idf = math.log((len(self.docmap) + 1) / (n_matching_docs + 1))
        return idf

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

    def load(self):

        if not os.path.exists(index_dir):
            raise FileNotFoundError(f"Could not find {index_dir}")

        if not os.path.exists(docmap_dir):
            raise FileNotFoundError(f"Could not find {docmap_dir}")

        if not os.path.exists(tf_dir):
            raise FileNotFoundError(f"Could not find {tf_dir}")

        with open(index_dir, "rb") as f:
            self.index = load(f)

        with open(docmap_dir, "rb") as f:
            self.docmap = load(f)

        with open(tf_dir, "rb") as f:
            self.term_frequencies = load(f)
