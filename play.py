#!/usr/bin/env python3
from flask import Flask, render_template, request, jsonify
import chess
import chess.svg

app = Flask(__name__)

# Initialize the chess board
board = chess.Board()


@app.route("/")
def index():
    return render_template("index.html", board_svg=chess.svg.board(board))


@app.route("/move", methods=["POST"])
def move():
    data = request.json
    mv = chess.Move.from_uci(data["move"])

    if mv in board.legal_moves:
        board.push(mv)
        return jsonify({"success": True, "board_svg": chess.svg.board(board)})
    else:
        return jsonify({"success": False, "message": "Invalid move"})


if __name__ == "__main__":
    app.run(debug=True)
