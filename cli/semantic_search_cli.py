import argparse

from cli.utils.semantic_search_parsers import register_parsers
from services.processing import load_movies
from services.semantic_search import (
    SemanticSearch,
    embed_query_text,
    embed_text,
    verify_embeddings,
    verify_model,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    register_parsers(parser)

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "embedquery":
            embed_query_text(args.query)

        case "verify_embeddings":
            verify_embeddings()

        case "search":
            documents = load_movies()

            ss = SemanticSearch()
            ss.load_or_create_embeddings(documents)
            results = ss.search(args.query, args.limit)
            for n, result in enumerate(results):
                title = result["title"].encode().decode("unicode-escape")

                print(f"{n + 1}. {title} (score: {result['score']:.4f}) ")

                words = result["description"].split()
                if len(words) > 20:
                    truncated_desc = " ".join(words[:20]) + "..."
                else:
                    truncated_desc = result["description"]

                print(f"{truncated_desc}\n")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
