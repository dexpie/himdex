from __future__ import annotations

import argparse
from pathlib import Path

import torch


def average_state_dicts(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
    second_weight: float,
) -> dict[str, torch.Tensor]:
    if first.keys() != second.keys():
        missing = sorted(set(first) ^ set(second))
        raise ValueError(f"Checkpoint state keys do not match: {missing[:5]}")
    if not 0.0 <= second_weight <= 1.0:
        raise ValueError("second_weight must be between 0 and 1")

    first_weight = 1.0 - second_weight
    averaged: dict[str, torch.Tensor] = {}
    for key, value in first.items():
        other = second[key]
        if value.shape != other.shape:
            raise ValueError(f"Shape mismatch for {key}: {value.shape} != {other.shape}")
        if torch.is_floating_point(value):
            averaged[key] = value * first_weight + other * second_weight
        else:
            averaged[key] = value
    return averaged


def average_checkpoints(
    first_path: str | Path,
    second_path: str | Path,
    output_path: str | Path,
    second_weight: float,
) -> None:
    first = torch.load(first_path, map_location="cpu")
    second = torch.load(second_path, map_location="cpu")
    averaged = dict(first)
    averaged["model_state"] = average_state_dicts(
        first["model_state"],
        second["model_state"],
        second_weight,
    )
    averaged["averaged_from"] = [
        {"checkpoint": str(first_path), "weight": 1.0 - second_weight},
        {"checkpoint": str(second_path), "weight": second_weight},
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(averaged, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average two compatible Himdex checkpoints.")
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--second-weight", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    average_checkpoints(
        args.first,
        args.second,
        args.output,
        args.second_weight,
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
