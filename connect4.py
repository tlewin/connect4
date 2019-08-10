import random

import numpy as np


BOARD_WIDTH = 7
BOARD_HEIGHT = 6
PLAYER1 = "O"
PLAYER2 = "X"


def new_board():
    return np.zeros([2, BOARD_HEIGHT, BOARD_WIDTH])


def clone_board(board):
    return np.copy(board)


def swap_players(board):
    return np.array([board[1], board[0]])


def print_board(board):

    def value2checker(value):
        if value == 1:
            return PLAYER1
        elif value == -1:
            return PLAYER2
        else:
            return " "

    for index in reversed(xrange(BOARD_HEIGHT)):
        row = board[0][index] - board[1][index]
        print "{}: {}".format(
            str(index + 1),
            " | ".join([value2checker(v) for v in row]),
        )


def play(board, player, column):
    return play_(clone_board(board), player, column)


def play_(board, player, column):
    if player not in [1, 2]:
        raise ValueError("Player must be 1 or 2: {}".format(player))

    row = BOARD_HEIGHT - 1
    if not (board[0][row][column] == 0 and board[1][row][column] == 0):
        raise ValueError("{} is already full.".format(column))

    row -= 1
    while row >= 0:
        if not (board[0][row][column] == 0 and board[1][row][column] == 0):
            break
        row -= 1

    board[player - 1][row + 1][column] = 1
    return board


def _has_winner(board, player):
    board = board[player - 1]

    # check all horizontals
    for r in xrange(BOARD_HEIGHT):
        for c in xrange(BOARD_WIDTH - 3):
            if board[r][c] == 0:
                continue
            if board[r][c] == board[r][c + 1] == board[r][c + 2] == board[r][c + 3]:
                return player

    # check all verticals
    for c in xrange(BOARD_WIDTH):
        for r in xrange(BOARD_HEIGHT - 3):
            if board[r][c] == 0:
                continue
            if board[r][c] == board[r + 1][c] == board[r + 2][c] == board[r + 3][c]:
                return player

    # check all diagonals
    for r in xrange(BOARD_HEIGHT - 3):
        for c in xrange(BOARD_WIDTH - 3):
            if board[r][c] == 0:
                continue
            if board[r][c] == board[r + 1][c + 1] == board[r + 2][c + 2] == board[r + 3][c + 3]:
                return player

    for r in xrange(BOARD_HEIGHT - 3):
        for c in xrange(BOARD_WIDTH - 1, 2, -1):
            if board[r][c] == 0:
                continue
            if board[r][c] == board[r + 1][c - 1] == board[r + 2][c - 2] == board[r + 3][c - 3]:
                return player

    return 0

def has_winner(board):
    return _has_winner(board, 1) or _has_winner(board, 2)


def available_plays(board):
    return [
        c
        for c in xrange(BOARD_WIDTH)
        if board[0][BOARD_HEIGHT - 1][c] == 0 and board[1][BOARD_HEIGHT - 1][c] == 0
    ]


def manual_player(board, player):
    while True:
        try:
            column = int(raw_input("Which column: ")) - 1
            if not column in available_plays(board):
                print "Invalid play"
                continue
            return column
        except ValueError:
            print "Please, enter a number between 1 and {}".format(BOARD_WIDTH)
            continue


def random_player(board, _):
    return random.choice(available_plays(board))


def play_match(player1_fn, player2_fn):
    board = new_board()
    player = random.choice([1, 2])
    player_fn = None
    while True:
        print_board(board)

        if not available_plays(board):
            print "No more available positions"
            return

        winner = has_winner(board)
        if not winner:
            print "Player {} turn:".format(player)
        else:
            print "Player {} won the match".format(1 if winner == 1 else 2)
            return

        player_fn = player1_fn if player == 1 else player2_fn
        column = player_fn(board, player)
        play_(board, player, column)
        player = 1 if player == 2 else 2


if __name__ == "__main__":
    play_match(manual_player, random_player)
