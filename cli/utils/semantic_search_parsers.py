import argparse


def register_parsers(parser: argparse.ArgumentParser):
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser(
        "verify",
        help="Print model information",
    )

    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Generate embeddings from text"
    )
    embed_text_parser.add_argument("text", type=str, help="The text to embed")

    subparsers.add_parser("verify_embeddings", help="Verify embeddings")

    embed_query_parser = subparsers.add_parser(
        "embedquery", help="Generate embeddings for query"
    )
    embed_query_parser.add_argument("query", type=str, help="The query to embed")
