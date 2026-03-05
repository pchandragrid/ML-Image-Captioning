import json
import random
from pathlib import Path
from typing import Iterable


def create_subset(
    input_path: str | Path,
    output_path: str | Path,
    size: int = 20_000,
) -> None:
    """
    Create a random subset of a JSONL annotations file.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open("r") as f:
        data = [json.loads(line) for line in f]

    if size > len(data):
        raise ValueError(f"Requested subset size {size} exceeds dataset size {len(data)}")

    subset = random.sample(data, size)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        for item in subset:
            f.write(json.dumps(item) + "\n")


def _main_from_cli(args: Iterable[str] | None = None) -> None:
    """
    Simple CLI wrapper when this module is executed as a script.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Create a random JSONL subset.")
    parser.add_argument(
        "--input",
        default="annotations/captions_train.jsonl",
        help="Input JSONL annotations path.",
    )
    parser.add_argument(
        "--output",
        default="annotations/subset_20k.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=20_000,
        help="Number of samples to keep.",
    )

    parsed = parser.parse_args(list(args) if args is not None else None)
    create_subset(parsed.input, parsed.output, parsed.size)
    print(f"Subset of {parsed.size} entries written to {parsed.output}")


if __name__ == "__main__":
    _main_from_cli()

