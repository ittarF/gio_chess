#!/usr/bin/env python3
import torch

class Engine():
    def __init__(self):
        from train import Net
        params = torch.load("models/model1310.pth")
        self.model = Net()
        self.model.load_state_dict(params)

    def value(self, brd):
        inp = brd.encode()
        with torch.no_grad():
            out = self.model(torch.tensor(inp).float().unsqueeze(0))
        return out
    
    def __call__(self, brd):
        legal_moves = brd.board.legal_moves
        if brd.board.turn:
            t = 1
        else:
            t = -1
        b_move, b_val = None, -100
        # evaluate each move
        for move in legal_moves:
            brd.board.push(move)
            val = self.value(brd)
            brd.board.pop()
            if val*t > b_val*t:
                b_move, b_val = move, val
        return b_move
        

if __name__ == "__main__":
    from board import Board
    engine = Engine()
    board = Board()
    print(engine(board))