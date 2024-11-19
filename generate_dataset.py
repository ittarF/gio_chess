#!/usr/bin/env python3
# to run: chmod +x generate_dataset.py
# to run: ./generate_dataset.py --path {file_name}.pgn --n 1000000 -o dataset_1M

import chess.pgn
import fire
import numpy as np
from board import Board


def get_dataset(path, num_samples=None, output_file="dataset"):
    """
    Parse a PGN file and return a dataset of board states and results.
    :param path: str, path to the PGN file
    :param num_samples: int, number of samples to extract from the PGN file
    :param output_file: str, name of the output file
    :return: tuple, (X, y) where X is a list of board states and y is a list of results

    Example:
    ./generate_dataset.py --path data/{file_name}.pgn --n 1000000 -o dataset_1M
    """
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

    X, y = np.array(X), np.array(y)

    np.savez(f"processed/{output_file}.npz", X, y)

    print(f"\nDataset saved to processed/{output_file}.npz")
    return


if __name__ == "__main__":

    fire.Fire(get_dataset)
