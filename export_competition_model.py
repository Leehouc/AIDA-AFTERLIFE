from __future__ import annotations

import argparse

from lhconnect6.model import export_competition_checkpoint, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a lightweight competition-compatible LHconnect6 checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Training checkpoint path.")
    parser.add_argument("--output", required=True, help="Output .pth path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    export_competition_checkpoint(
        model,
        args.output,
        extra={
            "exported_from": args.checkpoint,
            "epoch": checkpoint.get("epoch"),
        },
    )
    print(f"Exported competition checkpoint to {args.output}")


if __name__ == "__main__":
    main()
