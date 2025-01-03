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

    def __getattr__(self, name):
        return getattr(self.board, name)

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
        if self.board.ep_square is not None:
            assert b_state[self.board.ep_square] == 0
            b_state[self.board.ep_square] = 15

        b_state = b_state.reshape(8, 8)
        state = np.zeros((5, 8, 8), np.uint8)

        # binary state
        state[0] = (b_state >> 3) & 1
        state[1] = (b_state >> 2) & 1
        state[2] = (b_state >> 1) & 1
        state[3] = (b_state >> 0) & 1

        # turn
        state[4] = self.board.turn * 1.0

        return state

    def moves(self):
        return list(self.board.legal_moves)


if __name__ == "__main__":
    brd = Board()
    print(brd.encode())
    print(brd.moves())
