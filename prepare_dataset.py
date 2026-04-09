from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from lhconnect6.records import extract_supervised_samples, load_records, stack_samples
from lhconnect6.utils import dump_json, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LHconnect6 train/val NPZ datasets from replay JSON files.")
    parser.add_argument("--input", nargs="+", required=True, help="Replay file or directory containing JSON/JSONL records.")
    parser.add_argument("--output-dir", required=True, help="Output directory for train.npz, val.npz, and summary.json.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio after shuffling.")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed for sample shuffling.")
    parser.add_argument("--max-records", type=int, default=None, help="Optional record cap for quick experiments.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately on an invalid replay instead of skipping it.",
    )
    return parser.parse_args()


def save_split(path, samples: list[dict]) -> None:
    if not samples:
        return
    arrays = stack_samples(samples)
    np.savez_compressed(path, **arrays)


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    all_samples: list[dict] = []
    invalid_records: list[dict[str, str]] = []
    records_seen = 0
    record_count = 0
    source_counter: Counter[str] = Counter()
    stage_counter: Counter[int] = Counter()
    value_counter: Counter[str] = Counter()

    for file_path, record in load_records(args.input):
        records_seen += 1
        try:
            samples = extract_supervised_samples(record)
        except Exception as exc:
            invalid_records.append(
                {
                    "path": str(file_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if args.strict:
                raise
            continue
        if not samples:
            continue
        all_samples.extend(samples)
        record_count += 1
        for sample in samples:
            source_counter[sample["source"]] += 1
            stage_counter[int(sample["stage"])] += 1
            value_counter[str(int(sample["value_target"]))] += 1
        if args.max_records is not None and record_count >= args.max_records:
            break

    if not all_samples:
        if invalid_records:
            dump_json(output_dir / "invalid_records.json", invalid_records)
        raise RuntimeError("No training samples were extracted from the provided inputs.")

    rng = np.random.default_rng(args.seed)
    rng.shuffle(all_samples)

    if args.val_ratio <= 0 or len(all_samples) == 1:
        split_index = len(all_samples)
    else:
        split_index = int(round(len(all_samples) * (1.0 - args.val_ratio)))
        split_index = max(1, min(split_index, len(all_samples) - 1))

    train_samples = all_samples[:split_index]
    val_samples = all_samples[split_index:]

    save_split(output_dir / "train.npz", train_samples)
    if val_samples:
        save_split(output_dir / "val.npz", val_samples)

    summary = {
        "records_seen": records_seen,
        "records": record_count,
        "invalid_records": len(invalid_records),
        "samples_total": len(all_samples),
        "samples_train": len(train_samples),
        "samples_val": len(val_samples),
        "source_counts": dict(source_counter),
        "stage_counts": {str(key): int(value) for key, value in stage_counter.items()},
        "value_target_counts": dict(value_counter),
        "inputs": list(args.input),
    }
    dump_json(output_dir / "summary.json", summary)
    if invalid_records:
        dump_json(output_dir / "invalid_records.json", invalid_records)
    print(f"Saved dataset to {output_dir}")
    print(summary)
    if invalid_records:
        print(f"Skipped {len(invalid_records)} invalid record(s); details saved to {output_dir / 'invalid_records.json'}")


if __name__ == "__main__":
    main()
