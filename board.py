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
                    "p": 8,
                    "n": 9,
                    "b": 10,
                    "r": 11,
                    "q": 12,
                    "k": 13,
                }[piece.symbol()]
            
            if self.board.has_queenside_castling_rights(chess.WHITE):
                assert b_state[0] == 4
                b_state[0] = 7
            if self.board.has_kingside_castling_rights(chess.WHITE):
                assert b_state[7] == 4
                b_state[7] = 7
            if self.board.has_queenside_castling_rights(chess.BLACK):
                assert b_state[56] == 11
                b_state[56] = 14
            if self.board.has_kingside_castling_rights(chess.BLACK):
                assert b_state[63] == 11
                b_state[63] = 14             
            # todo: en passant and others?
        return b_state


if __name__ == "__main__":
    brd = Board()
    print(brd.encode())
