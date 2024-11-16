#!/usr/bin/env python3
# to run: chmod +x board.py
# to run: ./board.py
import chess


class Board:
    def __init__(self, board=None):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board

    def show(self):
        print(self.board)

    def encode(self):
        import numpy as np

        b_state = np.zeros((64), np.uint8)
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece is not None:
                b_state[i] = {
                    "P": 1,
                    "N": 2,
                    "B": 3,
                    "R": 4,
                    "Q": 5,
                    "K": 6,
                    "p": 9,
                    "n": 10,
                    "b": 11,
                    "r": 12,
                    "q": 13,
                    "k": 14,
                }[piece.symbol()]

        return b_state


if __name__ == "__main__":
    brd = Board()
    print(brd.encode())
