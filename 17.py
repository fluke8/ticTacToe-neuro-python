import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
from sklearn.preprocessing import StandardScaler
import pygame

def check_winner(board, player):
    # Проверка горизонтальных линий
    for row in board:
        if all(cell == player for cell in row):
            return True

    # Проверка вертикальных линий
    for col in range(len(board[0])):
        if all(row[col] == player for row in board):
            return True

    # Проверка диагоналей (основная и побочная)
    if all(board[i][i] == player for i in range(len(board))) or \
       all(board[i][len(board) - i - 1] == player for i in range(len(board))):
        return True

    return False

class ticTacToe():
    def __init__(self):
        self.positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

        self.input_size = 9
        self.output_size = 9

    def reset(self):

        self.num_steps = 0
        self.state =   [[0,0,0],
                        [0,0,0],
                        [0,0,0]]
        
        state_for_robots = scaler.transform(self.state)

        return state_for_robots.flatten()

    def step1(self, action):   
        done = False
        reward = 0
        self.num_steps += 1

        row, col = self.positions[action]

        if self.state[row][col] == 0:
            self.state[row][col] = 1
        else:
            done = True
            reward = -5 

        if not(np.isin(0, np.array(self.state))):
            done = True

        if check_winner(self.state, 1):
            done = True
            reward = 1
        
        state_for_robots = scaler.transform(self.state)


        
        return state_for_robots.flatten(), reward, done
    
    def step2(self, action):   
        done = False
        reward = 0
        self.num_steps += 1

        row, col = self.positions[action]

        if self.state[row][col] == 0:
            self.state[row][col] = -1
        else:
            done = True
            reward = -5  

        if not(np.isin(0, np.array(self.state))):
            done = True
            

        if check_winner(self.state,-1):
            done = True
            reward = 1
        

        state_for_robots = scaler.transform(self.state)
        
        return state_for_robots.flatten(), reward, done

    def render(self):
        for i in range(len(self.state)):
            print(self.state[i])

# Определение архитектуры нейронной сети для политики
class PolicyNetwork1(nn.Module):
    def __init__(self, input_size, hidden_size1, output_size):
        super(PolicyNetwork1, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc4 = nn.Linear(hidden_size1, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc4(x), dim=-1)  # Функция softmax для получения вероятностей действий

        return x
    
class PolicyNetwork2(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(PolicyNetwork2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size1, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.softmax(self.fc4(x), dim=-1)  # Функция softmax для получения вероятностей действий
        return x
    
# Функция для сэмплирования действия на основе политики
def select_action(net, state):
    state = torch.tensor(state, dtype=torch.float32)
    action_probs = net(state)
    action = np.random.choice(len(action_probs.detach().numpy()), p=action_probs.detach().numpy())
    return action

def optimize(net, optimizer, episode_states, episode_actions, episode_rewards):

    returns = []
    R = 0
    for r in episode_rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)

    policy_loss = []
    for i in range(len(episode_states)):
        action = episode_actions[i]
        G = returns[i]
        state = episode_states[i]
        state = np.array(state).flatten()
        action_prob = net(torch.tensor(state, dtype=torch.float32))[action]
        policy_loss.append(-torch.log(action_prob) * G)

    optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss).sum()

    policy_loss.backward()
    optimizer.step()

# Обучение агента
def train(net1, optimizer1, net2, optimizer2, episodes, hidden_size1):
    num_of_wins1, num_of_wins2 = 0, 0
    num_of_errors1, num_of_errors2 = 0, 0
    num_of_draws = 0
    for episode in range(episodes):
        print()
        print(f'Эпизод обучения номер {episode}, количество ничьих {num_of_draws}')
        print(f'Нейросеть, которая ходит первой. Количество побед: {num_of_wins1}, количество ошибок: {num_of_errors1}')
        print(f'Нейросеть, которая ходит второй. Количество побед: {num_of_wins2}, количество ошибок: {num_of_errors2}')

        state = env.reset()
       
        episode_states_net1 = []
        episode_actions_net1 = []
        episode_rewards_net1 = []

        episode_states_net2 = []
        episode_actions_net2 = []
        episode_rewards_net2 = []

        steps = 0
        while True:

            if steps%2==0:
                action = select_action(net1, state)
                next_state, reward, done = env.step1(action)
                if done:
                    if reward > 0:
                        num_of_wins1 += 1
                        episode_rewards_net2[len(episode_rewards_net2)-1] = -1
                        print('1 сеть выиграла')
                    elif reward < 0:
                        num_of_errors1 += 1
                    elif reward == 0:
                        num_of_draws += 1
                
                episode_states_net1.append(state)
                episode_actions_net1.append(action)
                episode_rewards_net1.append(reward)


            else:
                action = select_action(net2, state)
                next_state, reward, done = env.step2(action)
                if done:
                    if reward > 0:
                        num_of_wins2 += 1
                        episode_rewards_net1[len(episode_rewards_net1)-1] = -1
                    elif reward < 0:
                        num_of_errors2 += 1
                    elif reward == 0:
                        num_of_draws += 1

                episode_states_net2.append(state)
                episode_actions_net2.append(action)
                episode_rewards_net2.append(reward)


            
            steps += 1

            state = next_state
            
            if done:
                break
        
        print(episode_rewards_net1,episode_rewards_net2)
        if episode%500000 == 0:
            torch.save(net1, f'net/tictactoe/tictactoe_net1_{hidden_size1}_{episode}.pth')
            torch.save(net2, f'net/tictactoe/tictactoe_net2_{hidden_size1}_{episode}.pth')
        optimize(net1, optimizer1, episode_states_net1, episode_actions_net1, episode_rewards_net1)

        optimize(net2, optimizer2, episode_states_net2, episode_actions_net2, episode_rewards_net2)

        

def test1(net, num_episodes=10):
    for episode in range(num_episodes):
        print('RESTARTING')
        state = env.reset()
        step = 0
        while True:
            env.render()

            if step%2 == 0:
                action = int(input())
                next_state, reward, done = env.step1(action)
                if reward < 0:
                    print('Ты ошибся')
                elif reward > 0:
                    print('Ты выиграл')
            else:
                action = select_action(net, state)
                next_state, reward, done = env.step2(action)
                if reward < 0:
                    print('Робот ошибся')
                elif reward > 0:
                    print('Робот выиграл')
                
            step += 1

            if done:
                break

            state = next_state

def test2(net, num_episodes=10):
    for episode in range(num_episodes):
        print('RESTARTING')
        state = env.reset()
        step = 0
        while True:

            env.render()

            if step%2 == 0:
                action = select_action(net, state)
                next_state, reward, done = env.step1(action)
                if reward < 0:
                    print('Робот ошибся')
                elif reward > 0:
                    print('Робот выиграл')
            else:

                action = int(input())
                while True:
                    if action == 0 or action == 1 or action == 2 or action == 3 or action == 4 or action == 5 or action == 6 or action == 7 or action == 8:
                        next_state, reward, done = env.step2(action)
                        break
                    else:
                        print('неправильное действие')
                        action = int(input())

                if reward < 0:
                    print('Ты ошибся')
                elif reward > 0:
                    print('Ты выиграл')
            step += 1

            if done:
                break

            state = next_state

scaler = StandardScaler()

env = ticTacToe()

scaler.fit([[1,-1,0], [0,-1,0], [1,0,-1]])

episodes = 1000000000

input_size = env.input_size
output_size = env.output_size

hidden_size1_1 = 16

hidden_size1_2 = 42
hidden_size2_2 = 21
hidden_size3_2 = 21

learning_rate = 0.001
gamma = 0.99

net1 = PolicyNetwork1(input_size, hidden_size1_1, output_size)
optimizer1 = optim.Adam(net1.parameters(), lr=learning_rate)


net2 = PolicyNetwork1(input_size, hidden_size1_1, output_size)
optimizer2 = optim.Adam(net2.parameters(), lr=learning_rate)

render = False

train(net1, optimizer1, net2, optimizer2, episodes, hidden_size1_1)
print('Обучение закончено')


test1(net2)

test2(net1)
