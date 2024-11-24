#!/usr/bin/env python3
import torch


class Engine:
    def __init__(self):
        from models import ChessEvaluatorTransformer as Net

        params = torch.load("models/t_1301.pth")
        self.model = Net()
        self.model.load_state_dict(params)
        self.model.eval()

    def find_best(self, brd):
        """
        Given a board state return best move along with its evaluation
        """
        import numpy as np

        legal_moves = brd.legal_moves
        inp = []
        for move in legal_moves:
            brd.push(move)
            inp.append(brd.encode())
            brd.pop()
        with torch.no_grad():
            out = self.model(torch.tensor(np.array(inp)).float())

        b_move = None
        if brd.turn:
            t, b_val = 1, -100
        else:
            t, b_val = -1, 100

        for mv, val in zip(legal_moves, out):
            if t * val > t * b_val:
                b_val = val
                b_move = mv
        return b_move, b_val

    def minimax(self, brd, depth=1):

        # depth = 1

        b_val = 100
        b_move = None

        for move in brd.legal_moves:
            brd.push(move)
            _, val_oppo = self.find_best(brd)  # find best move of oppo (max)
            brd.pop()
            if val_oppo < b_val:
                b_move = move
                b_val = val_oppo

        return b_move, b_val


if __name__ == "__main__":
    from board import Board
    import time

    engine = Engine()
    board = Board()
    s = time.time()
    print(engine.minimax(board))
    print(time.time() - s)
