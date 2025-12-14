#!/usr/bin/env python3

import argparse
from dotenv import load_dotenv
from services.movies import load_json, search_field
import os

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            movies_path = os.environ["MOVIES_JSON"]
            movies = load_json(movies_path)["movies"]

            search_results = search_field(movies, field="title", query=args.query)
            search_results.sort(key=lambda x: x["id"])

            max_results = 5 if len(search_results) >= 5 else len(search_results)
            for i in range(max_results):
                print(f"{i+1}. {search_results[i]['title']}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
