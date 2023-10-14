import numpy as np

def check_winner(record_moves):
    board = np.zeros([3,3])
    player1 = 1
    player2 = -1

    for i, position in enumerate(record_moves):
        if i%2 == 0:
            board[position]=player1
        else:
            board[position]=player2

    for row in board:
        if all(cell == player1 for cell in row):
            return player1
        elif all(cell == player2 for cell in row):
            return player2

    for col in range(len(board[0])):
        if all(row[col] == player1 for row in board):
            return player1
        elif all(row[col] == player2 for row in board):
            return player2
        
    if all(board[i][i] == player1 for i in range(len(board))) or \
       all(board[i][len(board) - i - 1] == player1 for i in range(len(board))):
        return player1
    elif all(board[i][i] == player2 for i in range(len(board))) or \
       all(board[i][len(board) - i - 1] == player2 for i in range(len(board))):
        return player2

    return 0

class ticTacToe():
    def __init__(self):
        self.positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
        self.outcomes = []
    def evaluation_moves(self, record_moves, move):
        if move:
            record_moves.append(move)
        winner = check_winner(record_moves)
        if winner == 0 and len(record_moves) < 9:
            for position in self.positions:
                if not(position in record_moves):
                    self.evaluation_moves(record_moves.copy() , position)    
        else:
            self.outcomes.append([record_moves, winner])
    def step(self, move, num_step):
        # copy_outcomes = self.outcomes.copy()
        maked_moves = []
        for outcome in self.outcomes:
            if outcome[0][num_step] == move:
                maked_moves.append(outcome)
        self.outcomes = maked_moves
        print(self.outcomes)
        next_moves = []
        for position in self.positions:
            num_variable_steps = 0
            next_move = 0
            for outcome in self.outcomes:
                if outcome[0][num_step+1] == position:
                    num_variable_steps += 1
                    next_move += outcome[1]
            if next_move != 0 and num_variable_steps != 0:
                next_move = next_move / num_variable_steps
            next_moves.append(next_move)
        print(next_moves)

            
env = ticTacToe()

env.evaluation_moves([], None)

env.step((0,0), 0)
env.step((1,1), 1)
env.step((0,1), 2)
env.step((1,0), 3)
env.step((2,2), 4)



# print(env.outcomes)
