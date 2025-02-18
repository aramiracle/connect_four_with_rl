import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
from tqdm import tqdm
import random
from app.environment_train import ConnectFourEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Convolutional Dueling DQN model using PyTorch - Smaller Version with one more layer
class ConvDuelingDQN(nn.Module):
    def __init__(self):
        super(ConvDuelingDQN, self).__init__()
        # Convolutional layers for feature extraction - Reduced channels
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1) # Reduced to 16 channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # Reduced to 32 channels
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # Added conv3 with 64 channels

        # Value stream layers - Reduced size, Adjusted input size for FC layers
        self.fc_value = nn.Linear(6 * 7 * 64, 128) # Adjusted input size to 6*7*64 due to conv3
        self.fc1_value = nn.Linear(128, 32) # Reduced FC size
        self.fc2_value = nn.Linear(32, 1)

        # Advantage stream layers - Reduced size, Adjusted input size for FC layers
        self.fc_advantage = nn.Linear(6 * 7 * 64, 128) # Adjusted input size to 6*7*64 due to conv3
        self.fc1_advantage = nn.Linear(128, 32) # Reduced FC size
        self.fc2_advantage = nn.Linear(32, 7)

    def forward(self, x):
        x = x.long()
        x = F.one_hot(x.to(torch.int64), num_classes=3).float()
        x = x.unsqueeze(0) if len(x.size())==3 else x
        x = x.permute(0, 3, 1, 2) # Reshape to (batch_size, channels, width, height)

        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)) # Added conv3

        x = x.reshape(x.size(0), -1) # Flatten for FC layers

        # Value stream
        x_value = F.relu(self.fc_value(x))
        x_value = F.relu(self.fc1_value(x_value))
        value = self.fc2_value(x_value)

        # Advantage stream
        x_advantage = F.relu(self.fc_advantage(x))
        x_advantage = F.relu(self.fc1_advantage(x_advantage))
        advantage = self.fc2_advantage(x_advantage)

        # Combine value and advantage to get Q-values
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

# Implement experience replay buffer (No changes needed)
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

    def __len__(self):
        # Return the current size of the internal buffer
        return len(self.buffer)

# Pure Python is_instant_win and is_instant_loss - Modified to take player_piece
def is_instant_win(board, player_piece):
    piece = int(player_piece) # Ensure piece is int for comparison
    rows, cols = board.shape
    # Check horizontal
    for r in range(rows):
        for c in range(cols - 3):
            if board[r, c] == board[r, c + 1] == board[r, c + 2] == board[r, c + 3] == piece:
                return True
    # Check vertical
    for r in range(rows - 3):
        for c in range(cols):
            if board[r, c] == board[r + 1, c] == board[r + 2, c] == board[r + 3, c] == piece:
                return True
    # Check diagonals (top-left to bottom-right)
    for r in range(rows - 3):
        for c in range(cols - 3):
            if board[r, c] == board[r + 1, c + 1] == board[r + 2, c + 2] == board[r + 3, c + 3] == piece:
                return True
    # Check diagonals (top-right to bottom-left)
    for r in range(rows - 3):
        for c in range(3, cols):
            if board[r, c] == board[r + 1, c - 1] == board[r + 2, c - 2] == board[r + 3, c - 3] == piece:
                return True
    return False

def is_instant_loss(board, player_piece):
    opponent_piece = 3 - int(player_piece) # Opponent piece is 3 - player_piece (if player is 1, opponent is 2, if player is 2, opponent is 1)
    return is_instant_win(board, opponent_piece) # Loss for current player is win for opponent


# Define the DQN agent
class HybridAgent:
    def __init__(self, env, player_piece, buffer_capacity=1000000, batch_size=128, target_update_frequency=500, learning_rate=0.0001, instant_loss_penalty=1.0):
        self.env = env
        self.player_piece = player_piece # Store player piece
        self.model = ConvDuelingDQN().to(device)  # Using original ConvDuelingDQN model
        self.target_model = ConvDuelingDQN().to(device)  # Using original ConvDuelingDQN model
        self.target_model.load_state_dict(self.model.state_dict())
        self.buffer = ExperienceReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate) # Adjusted learning rate
        self.loss_fn = nn.MSELoss()
        self.num_training_steps = 0
        self.instant_loss_penalty_factor = instant_loss_penalty # Penalty factor is now a class attribute

    def select_action(self, state, epsilon):
        # Directly check which columns are not full
        available_actions = self.env.get_valid_actions()

        # Ensure the model is in evaluation mode
        self.model.eval()

        if random.random() < epsilon:
            action = random.choice(available_actions)
        else:
            # Check for instant win moves - now using player_piece
            instant_win_actions = [action for action in available_actions if self.is_instant_win(self.env, action)]
            if instant_win_actions:
                # If there are instant win moves, choose one randomly
                action = random.choice(instant_win_actions)
            else:
                # Filter out instant loss moves (Improved logic for both players)
                filtered_actions = []
                for action in available_actions:
                    if not self.is_instant_loss(self.env, action): # Check for *own* instant loss
                        temp_env = self.env.clone()
                        temp_env.step(action)
                        opponent_can_instant_win = False
                        for opponent_action in temp_env.get_valid_actions():
                            # Corrected call to self.is_instant_win to pass player_piece as positional argument
                            if self.is_instant_win(temp_env, opponent_action, 3 - self.player_piece): # Check if *opponent* can instant win after this action, using opponent's piece
                                opponent_can_instant_win = True
                                break
                        if not opponent_can_instant_win: # Only keep actions that do not lead to opponent instant win
                            filtered_actions.append(action)


                if filtered_actions:
                    # If there are filtered actions, choose the action with the highest Q-value among them
                    state_tensor = state.to(device)  # Adding batch dimension, move state to device
                    with torch.no_grad():
                        q_values = self.model(state_tensor).squeeze().cpu() # Move output back to CPU for action selection (argmax)

                    # Mask the Q-values of invalid actions with a very negative number
                    masked_q_values = torch.full(q_values.shape, float('-inf'))
                    masked_q_values[filtered_actions] = q_values[filtered_actions]

                    # Get the action with the highest Q-value among the valid actions
                    action = torch.argmax(masked_q_values).item()
                else:
                    # If there are no filtered actions (meaning all available actions lead to opponent instant win or are instant loss for self),
                    # choose a random action from all available actions to avoid getting stuck.
                    action = random.choice(available_actions)

        # Ensure the model is back in training mode
        self.model.train()

        return action

    def is_instant_win(self, env, action, player_piece=None): # Removed player_piece keyword argument from here, and added positional argument with default None
        if player_piece is None:
            player_piece = self.player_piece # Use instance's player_piece if not provided
        # Check if the agent (self.player_piece) has an instant winning move in the next turn
        next_env = env.clone()
        next_env.step(action)
        # Convert PyTorch tensor board to NumPy array before passing to Numba
        board_np = next_env.board.cpu().numpy()
        return is_instant_win(board_np, player_piece) # Pass player_piece to static is_instant_win

    def is_instant_loss(self, env, action):
        # Check if *I* (self.player_piece) have an instant losing move (instant win for opponent)
        next_env = env.clone()
        next_env.step(action)
        # Convert PyTorch tensor board to NumPy array before passing to Numba
        board_np = next_env.board.cpu().numpy()
        return is_instant_loss(board_np, self.player_piece) # Pass player_piece to static is_instant_loss

    def train_step(self):
        if len(self.buffer) >= self.batch_size:
            experiences = list(self.buffer.sample(self.batch_size))  # Convert to list for better indexing
            states, actions, rewards, next_states, dones = zip(*experiences)

            states = torch.stack(states).to(device) # Move states to device
            next_states = torch.stack(next_states).to(device) # Move next_states to device
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device) # Move rewards to device
            actions = torch.tensor(actions, dtype=torch.int64).to(device) # Move actions to device
            dones = torch.tensor(dones, dtype=torch.float32).to(device) # Move dones to device

            # Use target model for action selection in Double Q-learning
            target_actions = self.model(next_states).max(1)[1].unsqueeze(-1)
            max_next_q_values = self.target_model(next_states).gather(1, target_actions).squeeze(-1)

            current_q_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

            # Check if the chosen action is an instant loss move
            instant_loss_mask_list = []
            temp_env = ConnectFourEnv() # Need to create a new env for each state to check instant loss independently
            for i in range(len(states)):
                temp_env.reset(states[i].cpu().numpy()) # Set the board to the state from experience buffer, keep on CPU as env is likely CPU-based
                instant_loss_mask_list.append(self.is_instant_loss(temp_env, actions[i].item())) # is_instant_loss now player aware via class attribute
            instant_loss_mask = torch.tensor(instant_loss_mask_list, dtype=torch.float32).to(device) # Move mask to device


            # Introduce a penalty term for instant loss moves
            penalty = self.instant_loss_penalty_factor * instant_loss_mask

            # Calculate the expected Q values with the penalty
            expected_q_values = rewards + (1 - dones) * 0.99 * max_next_q_values - penalty  # Assuming a gamma of 0.99
            loss = self.loss_fn(current_q_values, expected_q_values)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.num_training_steps % self.target_update_frequency == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            self.num_training_steps += 1

def agent_vs_agent_train(agents, env, num_episodes=200000, epsilon_start=0.5, epsilon_final=0.01, epsilon_decay=0.9995): # Increased episodes, adjusted decay
    epsilon = epsilon_start
    reward_history_p1 = []
    reward_history_p2 = []

    for episode in tqdm(range(num_episodes), desc="Agent vs Agent Training", unit="episode"):
        state = env.reset()
        total_rewards = [0, 0]
        done = False

        player_turn = 1 # Keep track of player turn
        agent_index = 0 # Agent index corresponding to player turn

        while not done:
            agent = agents[agent_index] # Select agent based on player turn
            action = agent.select_action(state, epsilon)
            next_state, reward, done, info = env.step(action)
            total_rewards[agent_index] += reward
            agent.buffer.add(Experience(state, action, reward, next_state, done))
            state = next_state

            if done:
                break

            player_turn = 3 - player_turn # Switch player turn (1 -> 2, 2 -> 1)
            agent_index = 1 - agent_index # Switch agent index (0 -> 1, 1 -> 0)


        # Batch processing of experiences for each agent
        for agent in agents:
            agent.train_step()

        reward_history_p1.append(total_rewards[0])
        reward_history_p2.append(total_rewards[1])

        tqdm.write(f"Episode: {episode}, Winner: {info['winner']}, Total Reward Player 1: {total_rewards[0]}, Total Reward Player 2: {total_rewards[1]}, Epsilon: {epsilon:.2f}")

        if episode % 1000 == 0 and episode > 0:
            avg_reward_p1 = sum(reward_history_p1[-1000:]) / 1000
            avg_reward_p2 = sum(reward_history_p2[-1000:]) / 1000
            tqdm.write(f"--- Episode {episode} Report ---")
            tqdm.write(f"Average Reward over last 1000 episodes - Player 1: {avg_reward_p1:.2f}, Player 2: {avg_reward_p2:.2f}")
            tqdm.write(f"------------------------------")


        # Decay epsilon for the next episode
        epsilon = max(epsilon_final, epsilon * epsilon_decay)

    env.close()

# Example usage:
if __name__ == '__main__':
    env = ConnectFourEnv()

    # Players - using smaller model with one more layer - now passing player_piece
    hybrid_agents = [HybridAgent(env, player_piece=1, use_small_model=True), HybridAgent(env, player_piece=2, use_small_model=True)]

    # Agent vs Agent Training
    agent_vs_agent_train(hybrid_agents, env, num_episodes=200000) # Increased episodes

    # Save the trained agents - saving smaller agents with one more layer
    torch.save({
        'model_state_dict_player1': hybrid_agents[0].model.state_dict(),
        'target_model_state_dict_player1': hybrid_agents[0].target_model.state_dict(),
        'optimizer_state_dict_player1': hybrid_agents[0].optimizer.state_dict(),
        'model_state_dict_player2': hybrid_agents[1].model.state_dict(),
        'target_model_state_dict_player2': hybrid_agents[1].target_model.state_dict(),
        'optimizer_state_dict_player2': hybrid_agents[1].optimizer.state_dict(),
    }, 'saved_agents/hybrid_agents_after_train_player_aware_fixed_call.pth')