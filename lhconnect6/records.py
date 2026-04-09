from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from .board import Board
from .constants import BLACK, DRAW, VALUE_DRAW, VALUE_LOSS, VALUE_WIN, WHITE, flatten_xy


def normalize_turn(turn: dict[str, Any] | None) -> dict[str, int]:
    turn = turn or {}
    return {
        "x0": int(turn.get("x0", -1)),
        "y0": int(turn.get("y0", -1)),
        "x1": int(turn.get("x1", -1)),
        "y1": int(turn.get("y1", -1)),
    }


def _looks_like_botzone_replay_log(data: Any) -> bool:
    if not isinstance(data, list) or not data:
        return False
    has_request = False
    has_response = False
    for item in data:
        if not isinstance(item, dict):
            return False
        output = item.get("output")
        if isinstance(output, dict) and output.get("command") in {"request", "finish"}:
            has_request = True
        for player_key in ("0", "1"):
            player_item = item.get(player_key)
            if isinstance(player_item, dict) and "response" in player_item:
                has_response = True
    return has_request and has_response


def _extract_winner_from_botzone_log(data: list[dict[str, Any]]) -> int | None:
    for item in reversed(data):
        output = item.get("output")
        if not isinstance(output, dict) or output.get("command") != "finish":
            continue
        display = output.get("display")
        if isinstance(display, dict) and "winner" in display:
            parsed = parse_winner_value(display["winner"])
            if parsed is not None:
                return parsed
        content = output.get("content")
        if isinstance(content, dict):
            for nested_key in ("winner", "winplayer", "result"):
                if nested_key in content:
                    parsed = parse_winner_value(content[nested_key])
                    if parsed is not None:
                        return parsed
    return None


def _convert_botzone_replay_log(data: list[dict[str, Any]]) -> dict[str, Any]:
    move_sequence: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        for player_key in ("0", "1"):
            player_item = item.get(player_key)
            if isinstance(player_item, dict) and "response" in player_item:
                move = normalize_turn(player_item["response"])
                move_sequence.append(
                    {
                        "player": int(player_key),
                        **move,
                    }
                )
    return {
        "source_format": "botzone_replay_log",
        "move_sequence": move_sequence,
        "winner": _extract_winner_from_botzone_log(data),
    }


def infer_my_color(record: dict[str, Any]) -> int:
    requests = record.get("requests", [])
    if not requests:
        return BLACK
    first_request = normalize_turn(requests[0])
    return WHITE if first_request["x0"] != -1 and first_request["x1"] == -1 else BLACK


def parse_winner_value(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return BLACK if value else WHITE
    if isinstance(value, int):
        if value in (BLACK, WHITE, DRAW):
            return value
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"0", "black", "b"}:
            return BLACK
        if lowered in {"1", "white", "w"}:
            return WHITE
        if lowered in {"-1", "draw", "none", "unknown"}:
            return DRAW
    return None


def find_explicit_winner(record: dict[str, Any]) -> int | None:
    candidate_keys = [
        "winner",
        "winplayer",
        "result",
        "outcome",
        "label",
    ]
    for key in candidate_keys:
        if key not in record:
            continue
        value = record[key]
        if isinstance(value, dict):
            for nested_key in ("winner", "winplayer", "result"):
                if nested_key in value:
                    parsed = parse_winner_value(value[nested_key])
                    if parsed is not None:
                        return parsed
        else:
            parsed = parse_winner_value(value)
            if parsed is not None:
                return parsed
    return None


def build_board_from_record(record: dict[str, Any]) -> Board:
    if "move_sequence" in record:
        board = Board()
        for move in record["move_sequence"]:
            turn = normalize_turn(move)
            color = int(move["player"])
            if turn["x0"] != -1:
                board.play(color, turn["y0"], turn["x0"])
            if turn["x1"] != -1:
                board.play(color, turn["y1"], turn["x1"])
        return board

    requests = [normalize_turn(turn) for turn in record.get("requests", [])]
    responses = [normalize_turn(turn) for turn in record.get("responses", [])]
    my_color = infer_my_color(record)
    opp_color = 1 - my_color

    board = Board()
    for index in range(max(len(requests), len(responses))):
        if index < len(requests):
            turn = requests[index]
            if turn["x0"] != -1:
                board.play(opp_color, turn["y0"], turn["x0"])
            if turn["x1"] != -1:
                board.play(opp_color, turn["y1"], turn["x1"])
        if index < len(responses):
            turn = responses[index]
            if turn["x0"] != -1:
                board.play(my_color, turn["y0"], turn["x0"])
            if turn["x1"] != -1:
                board.play(my_color, turn["y1"], turn["x1"])
    return board


def infer_winner(record: dict[str, Any]) -> int:
    explicit = find_explicit_winner(record)
    if explicit is not None:
        return explicit
    board = build_board_from_record(record)
    winner = board.winner()
    return DRAW if winner is None else winner


def winner_to_value_target(winner: int, current_player: int) -> int:
    if winner == DRAW:
        return VALUE_DRAW
    return VALUE_WIN if winner == current_player else VALUE_LOSS


def extract_supervised_samples(record: dict[str, Any]) -> list[dict[str, Any]]:
    if "move_sequence" in record:
        winner = infer_winner(record)
        board = Board()
        samples: list[dict[str, Any]] = []
        move_index = 0

        def maybe_add(color: int, x: int, y: int, source: str, turn_index: int) -> None:
            nonlocal move_index
            if x < 0 or y < 0:
                return
            sample = {
                "input": board.get_nn_input(color),
                "policy_target": flatten_xy(x, y),
                "value_target": winner_to_value_target(winner, color),
                "player": color,
                "stage": board.stage,
                "move_index": move_index,
                "turn_index": turn_index,
                "source": source,
            }
            samples.append(sample)
            board.play(color, y, x)
            move_index += 1

        for turn_index, move in enumerate(record["move_sequence"]):
            turn = normalize_turn(move)
            color = int(move["player"])
            maybe_add(color, turn["x0"], turn["y0"], f"move0_p{color}", turn_index)
            maybe_add(color, turn["x1"], turn["y1"], f"move1_p{color}", turn_index)
        return samples

    requests = [normalize_turn(turn) for turn in record.get("requests", [])]
    responses = [normalize_turn(turn) for turn in record.get("responses", [])]
    my_color = infer_my_color(record)
    opp_color = 1 - my_color
    winner = infer_winner(record)

    board = Board()
    samples: list[dict[str, Any]] = []
    move_index = 0

    def maybe_add(color: int, x: int, y: int, source: str, turn_index: int) -> None:
        nonlocal move_index
        if x < 0 or y < 0:
            return
        sample = {
            "input": board.get_nn_input(color),
            "policy_target": flatten_xy(x, y),
            "value_target": winner_to_value_target(winner, color),
            "player": color,
            "stage": board.stage,
            "move_index": move_index,
            "turn_index": turn_index,
            "source": source,
        }
        samples.append(sample)
        board.play(color, y, x)
        move_index += 1

    for index in range(max(len(requests), len(responses))):
        if index < len(requests):
            turn = requests[index]
            maybe_add(opp_color, turn["x0"], turn["y0"], "request0", index)
            maybe_add(opp_color, turn["x1"], turn["y1"], "request1", index)
        if index < len(responses):
            turn = responses[index]
            maybe_add(my_color, turn["x0"], turn["y0"], "response0", index)
            maybe_add(my_color, turn["x1"], turn["y1"], "response1", index)
    return samples


def stack_samples(samples: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not samples:
        raise ValueError("no samples to stack")
    return {
        "inputs": np.stack([sample["input"] for sample in samples]).astype(np.float16),
        "policy_targets": np.asarray([sample["policy_target"] for sample in samples], dtype=np.int64),
        "value_targets": np.asarray([sample["value_target"] for sample in samples], dtype=np.int64),
        "players": np.asarray([sample["player"] for sample in samples], dtype=np.int8),
        "stages": np.asarray([sample["stage"] for sample in samples], dtype=np.int8),
        "move_indices": np.asarray([sample["move_index"] for sample in samples], dtype=np.int32),
        "turn_indices": np.asarray([sample["turn_index"] for sample in samples], dtype=np.int32),
    }


def _iter_json_objects_from_file(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                yield data
        return

    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        yield data
    elif isinstance(data, list):
        if _looks_like_botzone_replay_log(data):
            yield _convert_botzone_replay_log(data)
            return
        for item in data:
            if not isinstance(item, dict):
                raise ValueError(f"{path} contains a non-object record")
            yield item
    else:
        raise ValueError(f"{path} must contain an object or list of objects")


def iter_record_files(inputs: Iterable[str | Path]) -> Iterator[Path]:
    for raw_path in inputs:
        path = Path(raw_path)
        if path.is_dir():
            for file_path in sorted(path.rglob("*")):
                if file_path.suffix.lower() in {".json", ".jsonl"}:
                    yield file_path
        elif path.is_file():
            yield path
        else:
            raise FileNotFoundError(path)


def load_records(inputs: Iterable[str | Path]) -> Iterator[tuple[Path, dict[str, Any]]]:
    for file_path in iter_record_files(inputs):
        for record in _iter_json_objects_from_file(file_path):
            yield file_path, record
