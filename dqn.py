import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from collections import namedtuple, deque
from tqdm import tqdm

# Define the Connect Four environment using Gym

class ConnectFourEnv(gym.Env):
    def __init__(self):
        # Initialize the Connect Four board
        self.board = np.zeros((6, 7), dtype=np.float32)
        self.current_player = 1
        self.winner = None
        self.max_moves = 42  # Maximum number of moves in Connect Four
        self.action_space = gym.spaces.Discrete(7)
        self.observation_space = gym.spaces.Box(low=0, high=2, shape=(6, 7), dtype=np.float32)

    def reset(self):
        # Reset the environment to its initial state
        self.board = np.zeros((6, 7), dtype=np.float32)
        self.current_player = 1
        self.winner = None
        return self.board

    def step(self, action):
        # Check if game is already over
        if self.winner is not None:
            raise ValueError("Game is already over.")
        
        # Check if the action is valid (column is not full)
        if self.board[0][action] != 0:
            raise ValueError("Column is full.")

        # Apply the action
        row = self.get_next_open_row(action)
        self.board[row, action] = self.current_player

        # Check for a win
        if self.check_win(row, action):
            self.winner = self.current_player
            # The reward for winning or losing will be handled outside of the step function
            reward = 1 if self.current_player == 1 else -1
        elif np.count_nonzero(self.board) == self.max_moves:
            # The game is a draw
            self.winner = 0  # No winner in the case of a draw
            reward = 0
        else:
            # The game continues
            reward = 0

        # Prepare the return values
        done = self.winner is not None
        next_state = self.board.copy()
        info = {'winner': self.winner}

        # Switch players
        self.current_player = 3 - self.current_player

        return next_state, reward, done, info

    def render(self):
        # Print the current state of the board
        print(self.board)

    def get_next_open_row(self, col):
        for r in range(5, -1, -1):
            if self.board[r][col] == 0:
                return r

    def check_win(self, row, col):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, 4):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < 6 and 0 <= c < 7 and self.board[r, c] == self.board[row, col]:
                    count += 1
                else:
                    break
            for i in range(1, 4):
                r, c = row - dr * i, col - dc * i
                if 0 <= r < 6 and 0 <= c < 7 and self.board[r, c] == self.board[row, col]:
                    count += 1
                else:
                    break
            if count >= 4:
                return True
        return False

# Define the DQN model using PyTorch

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(6 * 7 * 3, 256)  # Multiply the input size by 2 for one-hot encoding
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 7)

    def forward(self, x):
        # One-hot encode the input state
        x = x.view(-1, 6 * 7)
        x = F.one_hot(x.to(torch.int64), num_classes=3).to(torch.float32)  # One-hot encode with 3 classes (0, 1, 2)
        x = x.view(-1, 6 * 7 * 3)  # Flatten the one-hot encoding
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)
# Implement experience replay buffer

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ExperienceReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        # Add an experience to the buffer
        self.buffer.append(experience)

    def sample(self, batch_size):
        # Sample a batch of experiences from the buffer
        return random.sample(self.buffer, batch_size)

# Define the DQN agent

class DQNAgent:
    def __init__(self, env, buffer_capacity=100000, batch_size=64, target_update_frequency=10):
        self.env = env
        self.model = DQN()
        self.target_model = DQN()
        self.target_model.load_state_dict(self.model.state_dict())
        self.buffer = ExperienceReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_fn = nn.MSELoss()

    def select_action(self, state, epsilon):
        # Update the environment state to match the provided state
        self.env.board = state

        # Select an action using an epsilon-greedy strategy
        available_columns = [col for col in range(7) if self.env.get_next_open_row(col) is not None]

        if not available_columns:
            return -1  # All columns are full, indicating a draw or game over

        if random.random() < epsilon:
            return random.choice(available_columns)

        state = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            q_values = self.model(state)

        available_q_values = [q_values[0][col] for col in available_columns]
        chosen_action = available_columns[torch.argmax(torch.stack(available_q_values)).item()]
        
        return chosen_action

    def train(self, num_episodes, epsilon_start=1.0, epsilon_final=0.05, epsilon_decay=0.9995):
        epsilon = epsilon_start
        player1_rewards = []
        player2_rewards = []

        for episode in tqdm(range(num_episodes)):  # Wrap the loop with tqdm for progress tracking
            state = self.env.reset()
            player1_total_reward = 0
            player2_total_reward = 0
            current_player = 1  # Start with player 1

            for step in range(self.env.max_moves):
                action = self.select_action(state, epsilon)

                next_state, reward, done, _ = self.env.step(action)
                
                # Check if the game has ended and assign rewards accordingly
                if done:
                    if self.env.winner == 1:
                        player1_reward = 1  # Win reward for player 1
                        player2_reward = -2 * player2_total_reward - 1  # Loss penalty for player 2
                        print('Player 1 wins')
                    elif self.env.winner == 2:
                        player1_reward = -2 * player1_total_reward - 1  # Loss penalty for player 1
                        player2_reward = 1  # Win reward for player 2
                        print('Player 2 wins')
                    else:  # It's a draw
                        player1_reward = -player1_total_reward
                        player2_reward = -player2_total_reward
                        print('Game is drawn!')
                else:
                    # Small negative reward for each move to encourage winning quickly
                    player1_reward = -1 / self.env.max_moves if current_player == 1 else 0
                    player2_reward = -1 / self.env.max_moves if current_player == 2 else 0

                player1_total_reward += player1_reward
                player2_total_reward += player2_reward

                self.buffer.add(Experience(state, action, reward, next_state, done))

                state = next_state

                if done:
                    break

                # Update model
                if len(self.buffer.buffer) >= self.batch_size:
                    experiences = self.buffer.sample(self.batch_size)
                    states, actions, rewards, next_states, dones = zip(*experiences)

                    states = np.array(states, dtype=np.single)
                    next_states = np.array(next_states, dtype=np.single)
                    rewards = np.array(rewards, dtype=np.single)
                    actions = np.array(actions, dtype=np.short)
                    dones = np.array(dones, dtype=np.single)

                    states = torch.tensor(states, dtype=torch.float)
                    next_states = torch.tensor(next_states, dtype=torch.float)
                    rewards = torch.tensor(rewards, dtype=torch.float)
                    actions = torch.tensor(actions, dtype=torch.int64)
                    dones = torch.tensor(dones, dtype=torch.float)

                    q_values = self.model(states)
                    with torch.no_grad():
                        next_q_values = self.target_model(next_states)
                        target_q_values = rewards + 0.99 * next_q_values.max(1)[0] * (1 - dones)

                    loss = self.loss_fn(q_values.gather(1, actions.view(-1, 1)), target_q_values.view(-1, 1))

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # Switch current player after each action
                current_player = 3 - current_player

            # Log progress for both players
            tqdm.write(f"Episode: {episode}, Player 1 Total Reward: {player1_total_reward:.4f}, Player 2 Total Reward: {player2_total_reward:.4f}, Epsilon: {epsilon:.2f}")

            epsilon = max(epsilon_final, epsilon * epsilon_decay)

        # Return the rewards for analysis if needed
        return player1_rewards, player2_rewards
    
# Main function
if __name__ == '__main__':
    env = ConnectFourEnv()
    dqn_agent = DQNAgent(env)

    # Train the DQN agent
    num_episodes = 3000
    dqn_agent.train(num_episodes=num_episodes)

    # Save the DQN agent's state after training
    torch.save({
        'model_state_dict': dqn_agent.model.state_dict(),
        'target_model_state_dict': dqn_agent.target_model.state_dict(),
        'optimizer_state_dict': dqn_agent.optimizer.state_dict(),
    }, 'saved_agents/dqn_agent_after_training.pth')
