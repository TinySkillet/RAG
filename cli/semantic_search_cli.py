import argparse

from cli.utils.semantic_search_parsers import register_parsers
from services.semantic_search import (
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

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
