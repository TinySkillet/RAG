import json
from typing import Any, List
from string import punctuation


def load_json(json_path: str) -> Any:
    with open(json_path, "r") as f:
        t = json.load(f)
        return t


def search_field(map_list: list[dict], field: str, query: str) -> List[dict]:
    search_results = []
    query = remove_punctuation(query)
    query_tks = tokenize(query)

    for hash_map in map_list:
        value = remove_punctuation(hash_map.get(field, ""))
        val_tks = tokenize(value)

        for q_tk in query_tks:
            if q_tk in val_tks:
                search_results.append(hash_map)

    return search_results


def remove_punctuation(value: str) -> str:
    value = value.lower()
    trans_table = str.maketrans("", "", punctuation)
    return value.translate(trans_table)


def tokenize(value: str) -> List[str]:
    tokens = value.split()
    return [token for token in tokens if token]
