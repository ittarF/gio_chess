#!/usr/bin/env python3
from flask import Flask, render_template, request, jsonify
import chess
import chess.svg
from board import Board
from engine import Engine  # Import the Engine class

app = Flask(__name__)

# Initialize the chess board and engine
board = Board()
engine = Engine()


def is_game_over():
    return (
        board.is_checkmate()
        or board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_seventyfive_moves()
        or board.is_fivefold_repetition()
    )


@app.route("/")
def index():
    return render_template("index.html", board_svg=chess.svg.board(board))


@app.route("/move", methods=["POST"])
def move():
    data = request.json
    mv = chess.Move.from_uci(data["move"])

    if mv in board.legal_moves:
        board.push(mv)
        if is_game_over():
            return jsonify(
                {
                    "success": True,
                    "board_svg": chess.svg.board(board),
                    "game_over": True,
                }
            )
        return jsonify({"success": True, "board_svg": chess.svg.board(board)})
    else:
        return jsonify({"success": False, "message": "Invalid move"})


@app.route("/undo", methods=["POST"])
def undo():
    if board.move_stack:
        board.pop()
        return jsonify({"success": True, "board_svg": chess.svg.board(board)})
    else:
        return jsonify({"success": False, "message": "No moves to undo"})


@app.route("/reset", methods=["POST"])
def reset():
    board.reset()
    return jsonify({"success": True, "board_svg": chess.svg.board(board)})


@app.route("/computer_move", methods=["POST"])
def computer_move():
    best_move = engine.minimax(board)[0]
    if best_move:
        board.push(best_move)
        if is_game_over():
            return jsonify(
                {
                    "success": True,
                    "board_svg": chess.svg.board(board),
                    "game_over": True,
                }
            )
        return jsonify({"success": True, "board_svg": chess.svg.board(board)})
    else:
        return jsonify({"success": False, "message": "No legal moves found"})


if __name__ == "__main__":
    app.run(debug=True)
