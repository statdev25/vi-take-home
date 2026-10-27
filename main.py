"""Entrypoint.

    uv run main.py train
    uv run main.py score
"""

import argparse

from assignment import pipeline


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="Fit on data/, write model artifacts to artifacts/.")
    sub.add_parser("score", help="Score patients, write out/targets.csv.")
    args = parser.parse_args()

    if args.command == "train":
        pipeline.train()
    else:
        pipeline.score()


if __name__ == "__main__":
    cli()
