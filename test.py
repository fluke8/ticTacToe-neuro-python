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
        

        # return state_for_robots.flatten()
        return np.array(self.state).flatten()

    def step1(self, action):   
        done = False
        reward = 0
        self.num_steps += 1

        row, col = self.positions[action]

        if self.state[row][col] == 0:
            self.state[row][col] = 1
        else:
            done = True
            reward = -3

        if not(np.isin(0, np.array(self.state))):
            done = True

        if check_winner(self.state, 1):
            done = True
            reward = 1
        


        
        # return state_for_robots.flatten(), reward, done
        return np.array(self.state).flatten(), reward, done
    
    def step2(self, action):   
        done = False
        reward = 0
        self.num_steps += 1

        row, col = self.positions[action]

        if self.state[row][col] == 0:
            self.state[row][col] = -1
        else:
            done = True
            reward = -3  

        if not(np.isin(0, np.array(self.state))):
            done = True
            

        if check_winner(self.state,-1):
            done = True
            reward = 1
        
        # return state_for_robots.flatten(), reward, done
        return np.array(self.state).flatten(), reward, done


    def render(self):
        for i in range(len(self.state)):
            print(self.state[i])

# Определение архитектуры нейронной сети для политики
class PolicyNetwork1(nn.Module):
    def __init__(self, input_size, hidden_size1, output_size):
        super(PolicyNetwork1, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        # dropout = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(hidden_size1, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc4(x), dim=-1)  # Функция softmax для получения вероятностей действий
        return x
    
class PolicyNetwork2(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(PolicyNetwork2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1) 
        print(x)  # Функция softmax для получения вероятностей действий
        return x

class PolicyNetwork3(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(PolicyNetwork3, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.softmax(self.fc4(x), dim=-1)
         # Функция softmax для получения вероятностей действий
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
def train(net1, optimizer1, net2, optimizer2, episodes, hidden_size1, hidden_size2, hidden_size3, learning_rate):
    num_of_wins1, num_of_wins2 = 0, 0
    num_of_errors1, num_of_errors2 = 0, 0
    num_of_draws = 0
    for episode in range(episodes):
        

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
                        episode_rewards_net2[len(episode_rewards_net2)-1] = -2
                    elif reward < 0:
                        num_of_errors1 += 1
                    elif reward == 0:
                        num_of_draws += 1
                        reward = -1
                        episode_rewards_net2[len(episode_rewards_net2)-1] = -1
                
                episode_states_net1.append(state)
                episode_actions_net1.append(action)
                episode_rewards_net1.append(reward)


            else:
                action = select_action(net2, state)
                next_state, reward, done = env.step2(action)
                if done:
                    if reward > 0:
                        num_of_wins2 += 1
                        episode_rewards_net1[len(episode_rewards_net1)-1] = -2
                    elif reward < 0:
                        num_of_errors2 += 1
                    elif reward == 0:
                        num_of_draws += 1
                        episode_rewards_net1[len(episode_rewards_net1)-1] = -1
                        reward = -1

                episode_states_net2.append(state)
                episode_actions_net2.append(action)
                episode_rewards_net2.append(reward)

            steps += 1

            state = next_state
            
            if done:
                break
        if episode%100==0:
            print(f'Эпизод обучения номер {episode}, количество ничьих {num_of_draws}')
            print(f'Нейросеть, которая ходит первой. Количество побед: {num_of_wins1}, количество ошибок: {num_of_errors1}')
            print(f'Нейросеть, которая ходит второй. Количество побед: {num_of_wins2}, количество ошибок: {num_of_errors2}')
            print(episode_rewards_net1,episode_rewards_net2)
            print(np.array(state).reshape(3, 3))

        if episode%300000 == 0:
            torch.save(net1, f'nets/tictactoe_net1_{hidden_size1}_{hidden_size2}_{hidden_size3}_{learning_rate}_{episode}.pth')
            torch.save(net2, f'nets/tictactoe_net2_{hidden_size1}_{hidden_size2}_{hidden_size3}_{learning_rate}_{episode}.pth')
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
                    action = int(input())-1
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

                action = int(input())-1
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


env = ticTacToe()




net1 = torch.load('nets/3tictactoe_net1_64_0_0_1e-07_2700000.pth')
net2 = torch.load('nets/3tictactoe_net2_64_0_0_1e-07_2700000.pth')



test2(net1)