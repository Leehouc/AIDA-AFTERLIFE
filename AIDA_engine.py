import json
import os

import numpy as np
import torch
import torch.nn as nn


BOARD_H = 15
BOARD_W = 15
INPUT_C = 43
DEFAULT_OPENING = (5, 7)
DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))


class CNNLayer(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(
            in_c,
            out_c,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)))


class ResnetLayer(nn.Module):
    def __init__(self, channels, mid_channels):
        super().__init__()
        self.conv1 = CNNLayer(channels, mid_channels)
        self.conv2 = CNNLayer(mid_channels, channels)

    def forward(self, x):
        return self.conv2(self.conv1(x)) + x


class OutputHeadResnet(nn.Module):
    def __init__(self, trunk_channels, head_mid_channels):
        super().__init__()
        self.cnn = CNNLayer(trunk_channels, head_mid_channels)
        self.valueHeadLinear = nn.Linear(head_mid_channels, 4)
        self.policyHeadLinear = nn.Conv2d(head_mid_channels, 1, kernel_size=1, bias=False)

    def forward(self, h):
        x = self.cnn(h)
        value = self.valueHeadLinear(x.mean((2, 3)))[:, :3]
        policy = self.policyHeadLinear(x).flatten(1)
        return value, policy


class ModelResNet(nn.Module):
    def __init__(self, blocks, channels):
        super().__init__()
        self.model_type = "res"
        self.model_param = (blocks, channels)
        self.inputhead = CNNLayer(INPUT_C, channels)
        self.trunk = nn.ModuleList([ResnetLayer(channels, channels) for _ in range(blocks)])
        self.outputhead = OutputHeadResnet(channels, channels)

    def forward(self, x):
        h = self.inputhead(x)
        for block in self.trunk:
            h = block(h)
        return self.outputhead(h)


MODEL_REGISTRY = {
    "res": ModelResNet,
}


class Board:
    OUT_OF_BOARD = 1314

    def __init__(self):
        self.board = np.zeros((2, BOARD_H, BOARD_W), dtype=np.float32)
        self.lastloc = (-1, -1)
        self.stage = 1

    def copy(self):
        clone = Board()
        clone.board = self.board.copy()
        clone.lastloc = self.lastloc
        clone.stage = self.stage
        return clone

    def play(self, color, y, x):
        if x < 0 or y < 0:
            return
        self.board[color, y, x] = 1.0
        if self.stage == 0:
            self.lastloc = (y, x)
        else:
            self.lastloc = (-1, -1)
        self.stage = 1 - self.stage

    def get_priority_value_array(self):
        if self.lastloc[0] == -1:
            return np.zeros((BOARD_H, BOARD_W), dtype=np.float32)

        stones = self.board[0] + self.board[1]
        total_weight = np.sum(stones)
        if total_weight == 0:
            return np.zeros((BOARD_H, BOARD_W), dtype=np.float32)

        y_coords, x_coords = np.meshgrid(np.arange(BOARD_H), np.arange(BOARD_W), indexing="ij")
        x_centroid = np.sum(x_coords * stones) / total_weight
        y_centroid = np.sum(y_coords * stones) / total_weight
        distances = (x_coords - x_centroid) ** 2 + (y_coords - y_centroid) ** 2
        return distances.astype(np.float32)

    def is_legal(self, x, y):
        if not (0 <= x < BOARD_W and 0 <= y < BOARD_H):
            return False
        if self.board[0, y, x] != 0 or self.board[1, y, x] != 0:
            return False
        if self.stage == 1 and self.lastloc[0] != -1:
            prv = self.get_priority_value_array()
            lastpr = prv[self.lastloc[0], self.lastloc[1]]
            if prv[y, x] < lastpr - 1e-10:
                return False
        return True

    def get_nn_input(self, next_player):
        nninput = np.zeros((INPUT_C, BOARD_H, BOARD_W), dtype=np.float32)
        if next_player == 0:
            nninput[0] = self.board[0]
            nninput[1] = self.board[1]
        else:
            nninput[0] = self.board[1]
            nninput[1] = self.board[0]

        if self.stage == 1:
            legal_moves = 1 - self.board[0] - self.board[1]
            if self.lastloc[0] != -1:
                nninput[2, self.lastloc[0], self.lastloc[1]] = 1.0
                prv = self.get_priority_value_array()
                lastpr = prv[self.lastloc[0], self.lastloc[1]]
                legal_moves = legal_moves * (prv >= lastpr - 1e-10)
            else:
                nninput[6] = 1.0
            nninput[3] = legal_moves
            nninput[4] = 1.0

        nninput[16] = -0.3
        return nninput

    def is_empty(self, x, y):
        return self.board[0, y, x] == 0 and self.board[1, y, x] == 0

    def iter_six_windows(self):
        for dx, dy in DIRECTIONS:
            for y in range(BOARD_H):
                for x in range(BOARD_W):
                    end_x = x + 5 * dx
                    end_y = y + 5 * dy
                    if 0 <= end_x < BOARD_W and 0 <= end_y < BOARD_H:
                        yield [(x + i * dx, y + i * dy) for i in range(6)]

    def get_relative_line(self, player, x, y, dx, dy):
        line = []
        opponent = 1 - player
        for step in range(-5, 6):
            nx = x + step * dx
            ny = y + step * dy
            if not (0 <= nx < BOARD_W and 0 <= ny < BOARD_H):
                line.append(self.OUT_OF_BOARD)
            elif step == 0:
                line.append(0)
            elif self.board[player, ny, nx] == 1:
                line.append(1)
            elif self.board[opponent, ny, nx] == 1:
                line.append(-1)
            else:
                line.append(0)
        return line

    def build_cpp_eval_line(self, player, x, y, direction_id):
        direction_map = {
            0: (1, 0),
            1: (0, 1),
            2: (1, -1),
            3: (1, 1),
        }
        dx, dy = direction_map[direction_id]
        return self.get_relative_line(player, x, y, dx, dy)

    @classmethod
    def apply_cpp_block_patterns(cls, line):
        score41 = 0
        score42 = 0
        if line[1] == cls.OUT_OF_BOARD or line[6] == cls.OUT_OF_BOARD:
            return score41, score42

        if line[0] == 0 and line[1] == -1 and line[2] == -1 and line[3] == -1 and line[4] == -1 and line[6] == 0:
            score41 += 1
        if line[1] == -1 and line[2] == -1 and line[3] == -1 and line[0] == -1 and line[4] == 0:
            score42 += 1
        if line[1] == -1 and line[2] == -1 and line[3] == -1 and line[4] == -1 and line[0] == 0 and line[6] != 0:
            score41 += 1
        if line[2] == -1 and line[3] == -1 and line[4] == -1 and line[6] == -1 and (line[1] == 0 or line[7] == 0):
            score41 += 1

        return score41, score42

    def score_cpp_block_move(self, player, x, y):
        score41 = 0
        score42 = 0
        for dx, dy in DIRECTIONS:
            line = self.get_relative_line(player, x, y, dx, dy)
            add41, add42 = self.apply_cpp_block_patterns(line)
            score41 += add41
            score42 += add42

            rev41, rev42 = self.apply_cpp_block_patterns(list(reversed(line)))
            score41 += rev41
            score42 += rev42
        return score41, score42

    def _apply_cpp_line_scan(self, line, x, y, direction_id, reverse, state):
        if state["mct_cout"]:
            return

        work = list(reversed(line)) if reverse else line
        if work[1] == self.OUT_OF_BOARD or work[6] == self.OUT_OF_BOARD:
            return

        if work[0] == 0 and work[1] == -1 and work[2] == -1 and work[3] == -1 and work[4] == -1 and work[6] == 0:
            state["grid41"][x, y] += 1
            state["grid_4_win"] = max(state["grid_4_win"], state["grid41"][x, y])
        if work[1] == -1 and work[2] == -1 and work[3] == -1 and work[0] == -1 and work[4] == 0:
            state["grid42"][x, y] += 1
            state["grid_4_win2"] = max(state["grid_4_win2"], state["grid42"][x, y])
        if work[1] == -1 and work[2] == -1 and work[3] == -1 and work[4] == -1 and work[0] == 0 and work[6] != 0:
            state["grid41"][x, y] += 1
            state["grid_4_win"] = max(state["grid_4_win"], state["grid41"][x, y])
        if work[2] == -1 and work[3] == -1 and work[4] == -1 and work[6] == -1 and (work[1] == 0 or work[7] == 0):
            state["grid41"][x, y] += 1
            state["grid_4_win"] = max(state["grid_4_win"], state["grid41"][x, y])

        if work[1] == 1 and work[2] == 1 and work[3] == 1 and work[4] == 1:
            if work[0] == 0:
                state["mct_cout"] = True
                if reverse:
                    second = self._cpp_endpoint(x, y, direction_id, 5)
                else:
                    second = self._cpp_endpoint(x, y, direction_id, -5)
                state["winning_pair"] = ((x, y), second)
                return
            if work[6] == 0:
                state["mct_cout"] = True
                if reverse:
                    second = self._cpp_endpoint(x, y, direction_id, -1)
                else:
                    second = self._cpp_endpoint(x, y, direction_id, 1)
                state["winning_pair"] = ((x, y), second)
                return

        if work[0] == 1 and work[1] == 1 and work[2] == 1 and work[3] == 1 and work[4] == 1:
            state["mct_cout5"] = True
            state["mct_cout"] = True
            state["winning_first"] = (x, y)

    def _cpp_endpoint(self, x, y, direction_id, step):
        if direction_id == 0:
            return x + step, y
        if direction_id == 1:
            return x, y + step
        if direction_id == 2:
            return x + step, y - step
        return x + step, y + step

    def _find_first_legal_blank_after(self, player, first_move):
        trial = self.copy()
        if not trial.is_legal(first_move[0], first_move[1]):
            return None
        trial.play(player, first_move[1], first_move[0])

        for x in range(BOARD_W):
            for y in range(BOARD_H):
                if not trial.is_empty(x, y):
                    continue
                if trial.is_legal(x, y):
                    return (x, y)
        return None

    def find_cpp_tactical_turn(self, player):
        if self.stage != 0:
            return None

        state = {
            "grid41": np.zeros((BOARD_W, BOARD_H), dtype=np.int32),
            "grid42": np.zeros((BOARD_W, BOARD_H), dtype=np.int32),
            "grid_4_win": 0,
            "grid_4_win2": 0,
            "mct_cout": False,
            "mct_cout5": False,
            "winning_pair": None,
            "winning_first": None,
        }

        for x in range(BOARD_W):
            for y in range(BOARD_H):
                if not self.is_empty(x, y):
                    continue

                for direction_id in (1, 0, 3, 2):
                    line = self.build_cpp_eval_line(player, x, y, direction_id)
                    self._apply_cpp_line_scan(line, x, y, direction_id, False, state)
                    self._apply_cpp_line_scan(line, x, y, direction_id, True, state)
                    if state["mct_cout"]:
                        break
                if state["mct_cout"]:
                    break
            if state["mct_cout"]:
                break

        if state["mct_cout"] and not state["mct_cout5"]:
            first, second = state["winning_pair"]
            trial = self.copy()
            if trial.is_legal(first[0], first[1]):
                trial.play(player, first[1], first[0])
                if trial.is_legal(second[0], second[1]):
                    return [first, second]

        if state["mct_cout"] and state["mct_cout5"]:
            first = state["winning_first"]
            second = self._find_first_legal_blank_after(player, first)
            if first is not None and second is not None:
                return [first, second]

        if state["grid_4_win"] >= 1:
            first = None
            for x in range(BOARD_W):
                for y in range(BOARD_H):
                    if self.is_empty(x, y) and self.is_legal(x, y) and state["grid41"][x, y] == state["grid_4_win"]:
                        first = (x, y)
                        break
                if first is not None:
                    break

            if first is not None:
                trial = self.copy()
                trial.play(player, first[1], first[0])
                second = None
                for x in range(BOARD_W):
                    for y in range(BOARD_H):
                        dist = max(abs(x - first[0]), abs(y - first[1]))
                        if dist < 4:
                            continue
                        if not trial.is_empty(x, y):
                            continue
                        if not trial.is_legal(x, y):
                            continue

                        if state["grid_4_win2"]:
                            if state["grid42"][x, y] == state["grid_4_win2"]:
                                second = (x, y)
                                break
                        elif state["grid41"][x, y]:
                            second = (x, y)
                            break
                    if second is not None:
                        break

                if second is not None:
                    return [first, second]

        return None

    def find_simple_win(self, player):
        opponent = 1 - player
        for cells in self.iter_six_windows():
            player_count = 0
            empty_positions = []
            blocked = False
            for nx, ny in cells:
                if self.board[opponent, ny, nx] == 1:
                    blocked = True
                    break
                if self.board[player, ny, nx] == 1:
                    player_count += 1
                else:
                    empty_positions.append((nx, ny))

            if blocked or not empty_positions:
                continue

            if player_count == 5 and len(empty_positions) == 1:
                nx, ny = empty_positions[0]
                if self.is_legal(nx, ny):
                    return nx, ny

        if self.stage == 0:
            for cells in self.iter_six_windows():
                player_count = 0
                empty_positions = []
                blocked = False
                for nx, ny in cells:
                    if self.board[opponent, ny, nx] == 1:
                        blocked = True
                        break
                    if self.board[player, ny, nx] == 1:
                        player_count += 1
                    else:
                        empty_positions.append((nx, ny))

                if blocked or player_count != 4 or len(empty_positions) != 2:
                    continue

                sequence = self.find_playable_sequence(player, empty_positions)
                if sequence is not None:
                    return sequence[0]
        return -1, -1

    def find_playable_sequence(self, player, moves):
        if not moves:
            return None
        if len(moves) == 1:
            x, y = moves[0]
            if self.is_legal(x, y):
                return [moves[0]]
            return None

        candidate_orders = [list(moves), list(reversed(moves))]
        for order in candidate_orders:
            trial = self.copy()
            valid = True
            for x, y in order:
                if not trial.is_legal(x, y):
                    valid = False
                    break
                trial.play(player, y, x)
            if valid:
                return order
        return None

    def collect_forcing_threats(self, player):
        opponent = 1 - player
        turn_start = self.copy()
        turn_start.stage = 0
        turn_start.lastloc = (-1, -1)
        threats = []

        for cells in self.iter_six_windows():
            player_count = 0
            empty_positions = []
            blocked = False
            for nx, ny in cells:
                if self.board[opponent, ny, nx] == 1:
                    blocked = True
                    break
                if self.board[player, ny, nx] == 1:
                    player_count += 1
                else:
                    empty_positions.append((nx, ny))

            if blocked:
                continue

            if player_count == 5 and len(empty_positions) == 1:
                threats.append({
                    "kind": "single",
                    "blocks": tuple(empty_positions),
                })
            elif player_count == 4 and len(empty_positions) == 2:
                if turn_start.find_playable_sequence(player, empty_positions) is not None:
                    threats.append({
                        "kind": "pair",
                        "blocks": tuple(empty_positions),
                    })

        return threats

    def find_must_block_move(self, player):
        opponent = 1 - player
        threats = self.collect_forcing_threats(opponent)
        if not threats:
            return -1, -1

        total_single = sum(1 for threat in threats if threat["kind"] == "single")
        center_x = (BOARD_W - 1) / 2.0
        center_y = (BOARD_H - 1) / 2.0
        best_move = (-1, -1)
        best_key = None

        for y in range(BOARD_H):
            for x in range(BOARD_W):
                if not self.is_legal(x, y):
                    continue

                covered_single = 0
                covered_total = 0
                for threat in threats:
                    if (x, y) in threat["blocks"]:
                        covered_total += 1
                        if threat["kind"] == "single":
                            covered_single += 1

                if covered_total == 0:
                    continue

                cpp41, cpp42 = self.score_cpp_block_move(player, x, y)
                trial = self.copy()
                trial.play(player, y, x)
                remaining = trial.collect_forcing_threats(opponent)
                remaining_single = sum(1 for threat in remaining if threat["kind"] == "single")
                distance = (x - center_x) ** 2 + (y - center_y) ** 2

                key = (
                    cpp41,
                    cpp42,
                    -remaining_single,
                    -len(remaining),
                    covered_single,
                    covered_total,
                    -distance,
                    -y,
                    -x,
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_move = (x, y)

        if best_move != (-1, -1):
            return best_move

        if total_single == 0:
            return -1, -1

        for threat in threats:
            if threat["kind"] != "single":
                continue
            x, y = threat["blocks"][0]
            if self.is_legal(x, y):
                return x, y

        return -1, -1


def load_model():
    project_root = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(project_root, "data", "con6_resnet_big.pth")
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    model_type = checkpoint.get("model_type", "res")
    model_param = checkpoint["model_param"]
    model = MODEL_REGISTRY[model_type](*model_param)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def read_payload():
    return json.loads(input())


def sample_action(policy_logits, board, move_count):
    policy_temp = 0.5 * (0.5 ** (move_count / 10)) + 0.01
    policy = policy_logits.detach().cpu().numpy().reshape((-1))
    policy = policy - np.max(policy)

    for idx in range(BOARD_W * BOARD_H):
        x = idx % BOARD_W
        y = idx // BOARD_W
        if not board.is_legal(x, y):
            policy[idx] = -10000

    policy = policy - np.max(policy)
    for idx in range(BOARD_W * BOARD_H):
        if policy[idx] < -1:
            policy[idx] = -10000

    probs = np.exp(policy / policy_temp)
    probs = probs / np.sum(probs)
    action = int(np.random.choice(np.arange(BOARD_W * BOARD_H), p=probs))
    return action % BOARD_W, action // BOARD_W


def main():
    model = load_model()
    full_input = read_payload()
    all_requests = full_input["requests"]
    all_responses = full_input["responses"]

    last_request = all_requests[-1]
    if last_request["x0"] == -1:
        print(json.dumps({
            "response": {
                "x0": DEFAULT_OPENING[0],
                "y0": DEFAULT_OPENING[1],
                "x1": -1,
                "y1": -1,
            }
        }))
        return

    next_player = 1 if (all_requests[0]["x0"] != -1 and all_requests[0]["x1"] == -1) else 0
    opp = 1 - next_player

    board = Board()
    for i in range(len(all_requests)):
        req = all_requests[i]
        if req["x0"] != -1:
            board.play(opp, req["y0"], req["x0"])
        if req["x1"] != -1:
            board.play(opp, req["y1"], req["x1"])

        if i < len(all_responses):
            resp = all_responses[i]
            if resp["x0"] != -1:
                board.play(next_player, resp["y0"], resp["x0"])
            if resp["x1"] != -1:
                board.play(next_player, resp["y1"], resp["x1"])

    tactical_moves = board.find_cpp_tactical_turn(next_player)
    if tactical_moves is not None:
        response = {
            "x0": tactical_moves[0][0],
            "y0": tactical_moves[0][1],
            "x1": tactical_moves[1][0],
            "y1": tactical_moves[1][1],
        }
        print(json.dumps({"response": response}))
        return

    my_action = {}
    for stage in range(2):
        assert board.stage == stage

        nninput = torch.from_numpy(board.get_nn_input(next_player)).unsqueeze(0)
        with torch.no_grad():
            _, policy_logits = model(nninput)

        action_x, action_y = sample_action(policy_logits, board, len(all_requests) * 2)
        my_action[f"x{stage}"] = action_x
        my_action[f"y{stage}"] = action_y
        board.play(next_player, action_y, action_x)

    print(json.dumps({"response": my_action}))


if __name__ == "__main__":
    main()
