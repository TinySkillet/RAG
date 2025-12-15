import json
from nltk import PorterStemmer
from typing import Any, List, Dict
from string import punctuation
import os
from dotenv import load_dotenv

load_dotenv()

movies_path = os.environ["MOVIES_JSON"]


def load_json(json_path: str) -> Any:
    with open(json_path, "r") as f:
        t = json.load(f)
        return t


def load_movies() -> List[Dict]:
    movies = load_json(movies_path)
    return movies["movies"]


def load_file(file_path: str) -> List[str]:
    with open(file_path, "r") as f:
        content = f.read()
        lines = content.splitlines()
        return lines


def search_field(field: str, query: str) -> List[dict]:
    map_list = load_movies()
    search_results = []
    query = remove_punctuation(query)
    query_tks = tokenize(query)

    for hash_map in map_list:
        value = remove_punctuation(hash_map.get(field, ""))
        val_tks = tokenize(value)

        for q_tk in query_tks:
            if q_tk in val_tks:
                search_results.append(hash_map)

    search_results.sort(key=lambda x: x["id"])
    return search_results


def remove_punctuation(value: str) -> str:
    value = value.lower()
    trans_table = str.maketrans("", "", punctuation)
    return value.translate(trans_table)


def tokenize(value: str) -> List[str]:
    tokens = value.split()
    removed = remove_stop_words(tokens)
    stemmed = stem(removed)
    return stemmed


def remove_stop_words(tokens: List[str]) -> List[str]:
    STOP_WORDS_PATH = os.environ["STOP_WORDS"]
    stop_words = load_file(STOP_WORDS_PATH)
    filtered = [token for token in tokens if token and token not in stop_words]
    return filtered


def stem(tokens: List[str]) -> List[str]:
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(token) for token in tokens]
    return stemmed
