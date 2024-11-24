#!/usr/bin/env python3
import torch
import numpy as np


class Engine:
    def __init__(self):
        from train import Net  # Import the neural network class

        # Load the trained model
        params = torch.load("models/model1310.pth")
        self.model = Net()
        self.model.load_state_dict(params)
        self.model.eval()

    def evaluate_board(self, brd):
        """
        Encode the board state and evaluate its score using the neural network.
        """
        with torch.no_grad():
            encoded_board = torch.tensor(brd.encode()).float().unsqueeze(0)
            return self.model(encoded_board).item()

    def find_best(self, brd):
        """
        Find the best move for the given board state using the neural network.
        """
        legal_moves = list(brd.legal_moves)  # Convert to list for reuse
        board_encodings = []

        # Generate board encodings for all legal moves
        for move in legal_moves:
            brd.push(move)
            board_encodings.append(brd.encode())
            brd.pop()

        # Batch process evaluations for efficiency
        with torch.no_grad():
            inputs = torch.tensor(np.array(board_encodings)).float()
            evaluations = self.model(inputs).squeeze(1)  # Shape: [num_moves]

        # Determine the best move
        is_white_turn = brd.turn
        t = 1 if is_white_turn else -1  # Turn multiplier
        best_value = float("-inf") if is_white_turn else float("inf")
        best_move = None

        for move, value in zip(legal_moves, evaluations):
            if t * value > t * best_value:
                best_value = value.item()
                best_move = move

        return best_move, best_value

    def minimax(self, brd, depth=3, alpha=float("-inf"), beta=float("inf")):
        """
        Perform minimax search with alpha-beta pruning to find the best move.
        """
        if depth == 0 or brd.is_game_over():
            return None, self.evaluate_board(brd)

        is_white_turn = brd.turn
        t = 1 if is_white_turn else -1  # Turn multiplier
        best_value = float("-inf") if is_white_turn else float("inf")
        best_move = None

        for move in brd.legal_moves:
            brd.push(move)
            _, value = self.minimax(brd, depth - 1, alpha, beta)
            brd.pop()

            if t * value > t * best_value:
                best_value = value
                best_move = move

            # Alpha-beta pruning
            if is_white_turn:
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break
            else:
                beta = min(beta, best_value)
                if beta <= alpha:
                    break

        return best_move, best_value


if __name__ == "__main__":
    from board import Board
    import time

    engine = Engine()
    board = Board()
    s = time.time()
    print(engine.minimax(board))
    print(time.time() - s)
