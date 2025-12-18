import argparse

from services.utils import BM25_B, BM25_K1


def register_parsers(parser: argparse.ArgumentParser):
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_keyword_parser = subparsers.add_parser(
        "search_keyword", help="Search movies using simple search"
    )
    search_keyword_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser(
        "search", help="Search movies using Inverted Index"
    )

    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Search movies using TF")
    tf_parser.add_argument("doc_id", type=int, help="Document id to search")
    tf_parser.add_argument("term", type=str, help="Term to search")

    idf_parser = subparsers.add_parser("idf", help="Search movies using IDF")
    idf_parser.add_argument("term", type=str, help="Term to search")

    tfidf_parser = subparsers.add_parser("tfidf", help="Search movies using TFIDF")
    tfidf_parser.add_argument("doc_id", type=int, help="Document id to search")
    tfidf_parser.add_argument("term", type=str, help="Term to search")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument("term", type=str, help="Term to search")

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document id and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document id to search")
    bm25_tf_parser.add_argument("term", type=str, help="Term to search")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Turntable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Turntable BM25 b parameter"
    )

    bm25_search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25_search_parser.add_argument("query", type=str, help="Search query")
    bm25_search_parser.add_argument(
        "--limit", type=int, nargs="?", default=5, help="Max results"
    )

    subparsers.add_parser("build", help="Build inverted indexes from movies list")
