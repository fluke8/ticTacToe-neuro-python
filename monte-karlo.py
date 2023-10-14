import numpy as np

def check_winner(board, player):
    for row in board:
        if all(cell == player for cell in row):
            return True

    for col in range(len(board[0])):
        if all(row[col] == player for row in board):
            return True
        
    if all(board[i][i] == player for i in range(len(board))) or \
       all(board[i][len(board) - i - 1] == player for i in range(len(board))):
        return True

    return False

class ticTacToe():
    def __init__(self):
        self.positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

        self.board = np.zeros(3, 3)
    
    def reset():
        board = np.zeros(3, 3)
        record_moves = []
        
    def step(board, player, record_moves):

