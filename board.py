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

    def __str__(self):
        return str(self.board)


if __name__ == "__main__":
    brd = Board()
    print(brd)
