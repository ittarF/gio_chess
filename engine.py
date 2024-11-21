#!/usr/bin/env python3
import torch


class Engine:
    def __init__(self):
        from train import Net

        params = torch.load("models/model1310.pth")
        self.model = Net()
        self.model.load_state_dict(params)
        self.model.eval()

    def find_best(self, brd):
        import numpy as np

        legal_moves = brd.board.legal_moves
        inp = []
        for move in legal_moves:
            brd.board.push(move)
            inp.append(brd.encode())
            brd.board.pop()
        with torch.no_grad():
            out = self.model(torch.tensor(np.array(inp)).float())

        b_move = None
        if brd.board.turn:
            t, b_val = 1, -100
        else:
            t, b_val = -1, 100

        for mv, val in zip(legal_moves, out):
            if t * val > t * b_val:
                b_val = val
                b_move = mv
        return b_move


if __name__ == "__main__":
    from board import Board

    engine = Engine()
    board = Board()
    print(engine.find_best(board))
