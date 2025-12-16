#!/usr/bin/env python3

import argparse

from services.indexing import InvertedIndex
from services.processing import search_field, tokenize
from services.utils import BM25_B, BM25_K1


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")

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

    args = parser.parse_args()
    match args.command:
        case "search_keyword":
            print(f"Searching for: {args.query}")
            search_results = search_field(field="title", query=args.query)

            max_results = 5 if len(search_results) >= 5 else len(search_results)
            for i in range(max_results):
                print(f"{i + 1}. {search_results[i]['title']}")

        case "search":
            inverted_index = InvertedIndex()
            try:
                inverted_index.load()
            except FileNotFoundError as e:
                print("Error:", e)
                exit(1)

            tokens = tokenize(args.query)
            matching_docs = inverted_index.search(tokens)
            print(f"Searching for: {args.query}")
            if not matching_docs:
                print(f"No results found for: {args.query}")
            else:
                for n, doc in enumerate(matching_docs):
                    print(f"{n + 1}. {doc['id']} {doc['title']}")

        case "bm25search":
            inverted_index = InvertedIndex()
            try:
                inverted_index.load()
            except FileNotFoundError as e:
                print("Error: ", e)
                exit(1)

            results = inverted_index.bm25_search(args.query, args.limit)
            for idx, id in enumerate(results):
                document = inverted_index.docmap[id]
                title = document["title"]
                score = results[id]
                print(f"{idx + 1}. ({id}) {title} - Score: {score:.2f}")

        case "bm25tf":
            inverted_index = InvertedIndex()
            try:
                inverted_index.load()
            except FileNotFoundError as e:
                print("Error: ", e)
                exit(1)

            bm25tf = inverted_index.get_bm25_tf(
                doc_id=args.doc_id, term=args.term, k1=args.k1, b=args.b
            )
            print(
                f"BM25 TF score of {args.term} in document {args.doc_id}: {bm25tf:.2f}"
            )

        case "bm25idf":
            inverted_index = InvertedIndex()
            try:
                inverted_index.load()
            except FileNotFoundError as e:
                print("Error: ", e)
                exit(1)

            bm25idf = inverted_index.get_bm25_idf(args.term)
            print(f"BM25 IDF score of {args.term}: {bm25idf:.2f}")

        case "tfidf":
            inverted_index = InvertedIndex()
            try:
                inverted_index.load()
            except FileNotFoundError as e:
                print("Error:", e)
                exit(1)

            tf = inverted_index.get_tf(doc_id=args.doc_id, term=args.term)
            idf = inverted_index.get_idf(args.term)

            tf_idf = tf * idf
            print(
                f"TF-IDF score of {args.term} in document {args.doc_id}: {tf_idf:.2f}"
            )

        case "tf":
            inverted_index = InvertedIndex()
            try:
                inverted_index.load()
            except FileNotFoundError as e:
                print("Error:", e)
                exit(1)

            term_freq = inverted_index.get_tf(doc_id=args.doc_id, term=args.term)
            print(
                f"Term frequency for term {args.term} in document {args.doc_id}: {term_freq}"
            )

        case "idf":
            inverted_index = InvertedIndex()
            try:
                inverted_index.load()
            except FileNotFoundError as e:
                print("Error:", e)
                exit(1)

            idf = inverted_index.get_idf(args.term)
            print(f"Inverse document frequency of {args.term}: {idf:.2f}")

        case "build":
            inverted_index = InvertedIndex()

            print("Building indexes...")
            inverted_index.build()

            print("Saving to disk...")
            inverted_index.save()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
