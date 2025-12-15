#!/usr/bin/env python3

import argparse
import math
from services.processing import search_field, tokenize
from services.indexing import InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_keyword_parser = subparsers.add_parser(
        "search_keyword", help="Search movies using BM25"
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

    subparsers.add_parser("build", help="Build inverted indexes from movies list")

    args = parser.parse_args()
    match args.command:
        case "search_keyword":
            print(f"Searching for: {args.query}")
            search_results = search_field(field="title", query=args.query)

            max_results = 5 if len(search_results) >= 5 else len(search_results)
            for i in range(max_results):
                print(f"{i+1}. {search_results[i]['title']}")

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
                    print(f"{n+1}. {doc['id']} {doc['title']}")

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

            tokens = tokenize(args.term)
            if len(tokens) > 1:
                raise ValueError("IDF can only be computed for a single token")

            token = tokens[0]
            n_matching_docs = len(inverted_index.get_documents(token))
            idf = math.log((len(inverted_index.docmap) + 1) / (n_matching_docs + 1))
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
