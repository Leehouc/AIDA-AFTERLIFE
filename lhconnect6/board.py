from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import BLACK, BOARD_HEIGHT, BOARD_WIDTH, INPUT_CHANNELS


@dataclass(frozen=True)
class Move:
    color: int
    x: int
    y: int


class Board:
    """
    Lightweight 15x15 Connect6 board matching the stage semantics in comptition.py.

    stage == 1:
        - initial single-stone opening, or
        - the second stone of a normal two-stone turn
    stage == 0:
        - the first stone of a normal two-stone turn
    """

    def __init__(self) -> None:
        self.board = np.zeros((2, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
        self.lastloc = (-1, -1)
        self.stage = 1

    def clone(self) -> "Board":
        other = Board()
        other.board = self.board.copy()
        other.lastloc = self.lastloc
        other.stage = self.stage
        return other

    def stone_count(self) -> int:
        return int(np.sum(self.board))

    def occupied(self) -> np.ndarray:
        return self.board[0] + self.board[1]

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT

    def play(self, color: int, y: int, x: int) -> None:
        if x < 0 or y < 0:
            return
        if not self.in_bounds(x, y):
            raise ValueError(f"move out of bounds: {(x, y)}")
        if self.occupied()[y, x] != 0:
            raise ValueError(f"illegal occupied move: {(x, y)}")

        self.board[color, y, x] = 1.0
        if self.stage == 0:
            self.lastloc = (y, x)
        else:
            self.lastloc = (-1, -1)
        self.stage = 1 - self.stage

    def get_priority_value_array(self) -> np.ndarray:
        if self.lastloc[0] == -1:
            return np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)

        stones = self.occupied()
        total_weight = np.sum(stones)
        if total_weight == 0:
            return np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)

        y_coords, x_coords = np.meshgrid(
            np.arange(BOARD_HEIGHT),
            np.arange(BOARD_WIDTH),
            indexing="ij",
        )
        x_centroid = np.sum(x_coords * stones) / total_weight
        y_centroid = np.sum(y_coords * stones) / total_weight
        distances = (x_coords - x_centroid) ** 2 + (y_coords - y_centroid) ** 2
        return distances.astype(np.float32)

    def legal_mask(self, strict_priority: bool = True) -> np.ndarray:
        legal = 1.0 - self.occupied()
        if strict_priority and self.stage == 1 and self.lastloc[0] != -1:
            priority_values = self.get_priority_value_array()
            last_priority = priority_values[self.lastloc[0], self.lastloc[1]]
            legal = legal * (priority_values >= last_priority - 1e-10)
        return legal.astype(np.float32)

    def is_legal(self, x: int, y: int, strict_priority: bool = True) -> bool:
        if not self.in_bounds(x, y):
            return False
        if self.occupied()[y, x] != 0:
            return False
        if strict_priority and self.stage == 1 and self.lastloc[0] != -1:
            priority_values = self.get_priority_value_array()
            last_priority = priority_values[self.lastloc[0], self.lastloc[1]]
            if priority_values[y, x] < last_priority - 1e-10:
                return False
        return True

    def get_nn_input(self, next_player: int) -> np.ndarray:
        nninput = np.zeros((INPUT_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
        if next_player == BLACK:
            nninput[0] = self.board[0]
            nninput[1] = self.board[1]
        else:
            nninput[0] = self.board[1]
            nninput[1] = self.board[0]

        if self.stage == 1:
            legal_moves = self.legal_mask(strict_priority=True)
            if self.lastloc[0] != -1:
                nninput[2, self.lastloc[0], self.lastloc[1]] = 1.0
            else:
                nninput[6] = 1.0
            nninput[3] = legal_moves
            nninput[4] = 1.0

        nninput[16] = -0.3
        return nninput

    def legal_moves(self, strict_priority: bool = True) -> list[tuple[int, int]]:
        legal = self.legal_mask(strict_priority=strict_priority)
        ys, xs = np.where(legal > 0.5)
        return [(int(x), int(y)) for y, x in zip(ys, xs)]

    def find_simple_win(self, player: int, strict_legal: bool = True) -> tuple[int, int]:
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                for dx, dy in directions:
                    count_player = 0
                    empty_positions: list[tuple[int, int]] = []
                    valid_window = True
                    for i in range(6):
                        nx = x + i * dx
                        ny = y + i * dy
                        if not self.in_bounds(nx, ny):
                            valid_window = False
                            break
                        if self.board[player, ny, nx] == 1:
                            count_player += 1
                        elif self.board[0, ny, nx] == 0 and self.board[1, ny, nx] == 0:
                            empty_positions.append((nx, ny))

                    if not valid_window or not empty_positions:
                        continue

                    can_finish = (
                        (self.stage == 0 and count_player == 4 and len(empty_positions) == 2)
                        or (self.stage == 0 and count_player == 5 and len(empty_positions) == 1)
                        or (self.stage == 1 and count_player == 5 and len(empty_positions) == 1)
                    )
                    if not can_finish:
                        continue

                    for nx, ny in empty_positions:
                        if not strict_legal or self.is_legal(nx, ny, strict_priority=True):
                            return nx, ny
        return (-1, -1)

    def winner(self) -> int | None:
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for color in (0, 1):
            for y in range(BOARD_HEIGHT):
                for x in range(BOARD_WIDTH):
                    if self.board[color, y, x] != 1:
                        continue
                    for dx, dy in directions:
                        ok = True
                        for i in range(1, 6):
                            nx = x + dx * i
                            ny = y + dy * i
                            if not self.in_bounds(nx, ny) or self.board[color, ny, nx] != 1:
                                ok = False
                                break
                        if ok:
                            return color
        return None
