from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch

from .board import Board
from .constants import DEFAULT_OPENING, unflatten_index
from .records import build_board_from_record, infer_my_color


def reconstruct_payload_state(payload: dict[str, Any]) -> tuple[Board, int]:
    board = build_board_from_record(payload)
    next_player = infer_my_color(payload)
    return board, next_player


def _masked_policy_logits(
    board: Board,
    next_player: int,
    model: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    nninput = board.get_nn_input(next_player)
    input_tensor = torch.from_numpy(nninput).unsqueeze(0).to(device)
    with torch.no_grad():
        _, policy_logits = model(input_tensor)
    logits = policy_logits[0].detach().cpu().numpy().astype(np.float64)
    legal_mask = board.legal_mask(strict_priority=True).reshape(-1)
    logits[legal_mask <= 0.5] = -1e9
    return logits


def choose_move(
    board: Board,
    next_player: int,
    model: torch.nn.Module,
    device: torch.device,
    temperature: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[int, int]:
    winloc = board.find_simple_win(next_player, strict_legal=True)
    if winloc[0] != -1:
        return winloc

    logits = _masked_policy_logits(board, next_player, model, device)
    legal_indices = np.flatnonzero(logits > -1e8)
    if legal_indices.size == 0:
        raise RuntimeError("no legal moves available after masking")

    if temperature <= 0:
        action = int(np.argmax(logits))
        return unflatten_index(action)

    logits = logits - np.max(logits[legal_indices])
    scaled = logits / temperature
    scaled = scaled - np.max(scaled[legal_indices])
    probs = np.exp(scaled)
    probs[np.logical_not(np.isfinite(probs))] = 0.0
    probs[logits <= -1e8] = 0.0
    total = probs.sum()
    if total <= 0:
        action = int(np.argmax(logits))
        return unflatten_index(action)
    probs /= total
    if rng is None:
        rng = np.random.default_rng()
    action = int(rng.choice(np.arange(probs.shape[0]), p=probs))
    return unflatten_index(action)


def choose_turn(
    payload: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    temperature: float = 0.0,
    opening: tuple[int, int] = DEFAULT_OPENING,
    rng: np.random.Generator | None = None,
) -> dict[str, int]:
    all_requests = payload.get("requests", [])
    if not all_requests:
        raise ValueError("payload must contain requests")

    last_request = all_requests[-1]
    if int(last_request.get("x0", -1)) == -1:
        return {
            "x0": opening[0],
            "y0": opening[1],
            "x1": -1,
            "y1": -1,
        }

    board, next_player = reconstruct_payload_state(payload)
    stones_to_play = 1 if board.stage == 1 else 2
    response = {"x0": -1, "y0": -1, "x1": -1, "y1": -1}
    for move_slot in range(stones_to_play):
        x, y = choose_move(
            board=board,
            next_player=next_player,
            model=model,
            device=device,
            temperature=temperature,
            rng=rng,
        )
        response[f"x{move_slot}"] = int(x)
        response[f"y{move_slot}"] = int(y)
        board.play(next_player, y, x)
    return response


def run_payload_json(
    raw_payload: str,
    model: torch.nn.Module,
    device: torch.device,
    temperature: float = 0.0,
    opening: tuple[int, int] = DEFAULT_OPENING,
    rng: np.random.Generator | None = None,
) -> str:
    payload = json.loads(raw_payload.lstrip("\ufeff"))
    response = choose_turn(
        payload,
        model=model,
        device=device,
        temperature=temperature,
        opening=opening,
        rng=rng,
    )
    return json.dumps({"response": response})
