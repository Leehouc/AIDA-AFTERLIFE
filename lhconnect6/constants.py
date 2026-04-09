BOARD_SIZE = 15
BOARD_HEIGHT = 15
BOARD_WIDTH = 15
INPUT_CHANNELS = 43

BLACK = 0
WHITE = 1
DRAW = -1

VALUE_LOSS = 0
VALUE_DRAW = 1
VALUE_WIN = 2

DEFAULT_MODEL_BLOCKS = 20
DEFAULT_MODEL_CHANNELS = 128
DEFAULT_OPENING = (5, 7)


def flatten_xy(x: int, y: int) -> int:
    return y * BOARD_WIDTH + x


def unflatten_index(index: int) -> tuple[int, int]:
    return index % BOARD_WIDTH, index // BOARD_WIDTH
