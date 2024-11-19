#!/usr/bin/env python3
# to run: chmod +x generate_dataset.py
# to run: ./generate_dataset.py

import chess.pgn
#import numpy as np
from board import Board


def get_dataset(num_samples=None, path="data/lichess_db_standard_rated_2013-01.pgn"):
    X, y = [], []
    values = {"1/2-1/2": 0, "0-1": -1, "1-0": 1}
    gn = 0
    
    with open(path, "r", encoding="utf-8") as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            board = Board()
            for move in game.mainline_moves():
                board.board.push(move)
                X.append(board.encode())
                y.append(values[game.headers["Result"]])
            gn += 1    
            print(f"Game {gn:8} parsed | total samples: {len(X)}", end="\r")        
            if num_samples is not None and len(X) >= num_samples:
                break

    return X, y


if __name__ == "__main__":

    X_a, y_a = get_dataset()
    print(len(X_a), len(y_a))