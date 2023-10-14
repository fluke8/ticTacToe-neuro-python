import random

def get_empty_cells(board):
    empty_cells = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                empty_cells.append((i, j))
    return empty_cells

def simulate_game(board, player):
    empty_cells = get_empty_cells(board)
    if len(empty_cells) == 0:
        return 0  # Ничья
    random_empty_cell = random.choice(empty_cells)
    row, col = random_empty_cell
    board[row][col] = player
    if check_winner(board, player):
        return 1  # Игрок выиграл
    else:
        return -simulate_game(board, 'X' if player == 'O' else 'O')

def check_winner(board, player):
    # Проверка выигрышных комбинаций
    for i in range(3):
        if all(board[i][j] == player for j in range(3)) or \
           all(board[j][i] == player for j in range(3)):
            return True
    if all(board[i][i] == player for i in range(3)) or \
       all(board[i][2 - i] == player for i in range(3)):
        return True
    return False

def monte_carlo(board, player, simulations):
    scores = {move: 0 for move in get_empty_cells(board)}
    for _ in range(simulations):
        for move in scores:
            simulated_board = [row[:] for row in board]  # Создаем копию доски
            row, col = move
            simulated_board[row][col] = player
            scores[move] += simulate_game(simulated_board, 'X' if player == 'O' else 'O')
    best_move = max(scores, key=scores.get)
    return best_move

# Пример использования
if __name__ == "__main__":
    board = [[' ' for _ in range(3)] for _ in range(3)]
    player = 'X'
    simulations = 10000

    for _ in range(4):  # Сделаем 4 хода
        row, col = monte_carlo(board, player, simulations)
        board[row][col] = player
        print(f"Игрок {player} сделал ход в ({row}, {col})")
        player = 'X' if player == 'O' else 'O'

    # Отобразим итоговое состояние доски
    for row in board:
        print(" ".join(row))
