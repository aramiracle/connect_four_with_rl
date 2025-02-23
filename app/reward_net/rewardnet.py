import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
import time
import random
from tqdm import tqdm
from app.environment_test import ConnectFourEnv  # Assuming ConnectFourEnv is in app.environment_test

# --- Hyperparameters ---
DATASET_NAME = "connect_four_balanced"  # Increment dataset name
MODEL_NAME = "connect_four_reward_net"  # Increment model name
DATASET_PATH = f"{DATASET_NAME}.pth"
MODEL_SAVE_PATH = f"{MODEL_NAME}.pth"

NUM_STATES_PER_OUTCOME = 50000  # Target states per outcome (win/loss)
DATA_GENERATION_MAIN_AGENT_DEPTH = 3
INITIAL_OPPONENT_DEPTH = 4
OPPONENT_DEPTH_ADJUST_FREQUENCY = 100
OPPONENT_DEPTH_ADJUST_RATIO = 1.25
MIN_OPPONENT_DEPTH = 2
MAX_OPPONENT_DEPTH = 4  # Optional max depth to prevent excessive depth
MIN_BACKWARD_DEPTH_TO_TERMINAL_SAVE = 1

REWARD_NETWORK_EPOCHS = 100
REWARD_NETWORK_BATCH_SIZE = 1024
REWARD_NETWORK_LEARNING_RATE = 0.0005
REWARD_NETWORK_PRINT_FREQUENCY = 10

REPORT_INTERVAL = 10  # Time interval (in seconds) for process reporting

class AlphaBetaAgent:
    def __init__(self, env, depth=4, player=1, data_generation_mode=False, data_buffer=None):
        self.env = env
        self.depth = depth
        self.player = player
        self.data_generation_mode = data_generation_mode
        self.data_buffer = data_buffer
        self.transposition_table = {} # Transposition table for memoization

    def alpha_beta(self, current_env, depth, alpha, beta, maximizing_player, current_depth_in_search=0): # Added current_depth_in_search and return depth_to_terminal
        state_tuple = tuple(current_env.board.flatten().tolist()) # Convert state to hashable tuple

        # Check transposition table
        if state_tuple in self.transposition_table and self.transposition_table[state_tuple][1] >= depth:
            return self.transposition_table[state_tuple][0], self.transposition_table[state_tuple][2] # Return value and depth_to_terminal

        if depth == 0 or current_env.is_terminal():
            if current_env.is_terminal():
                if current_env.winner == self.player if maximizing_player else (3 - self.player):
                    terminal_value = float('inf')
                elif current_env.winner == (3 - self.player) if maximizing_player else self.player:
                    terminal_value = float('-inf')
                else:
                    terminal_value = 0 # Draw
                # No saving of terminal states here anymore
                self.transposition_table[state_tuple] = (terminal_value, depth, 0) # Store in transposition table, depth_to_terminal = 0 for terminal state
                return terminal_value, 0 # depth to terminal is 0 for terminal states
            else:
                value = 0 # No heuristic, return 0 for non-terminal depth limit
                self.transposition_table[state_tuple] = (value, depth, 0) # Store in transposition table, depth_to_terminal = 0 for depth limit reached
                return value, 0 # depth to terminal is 0 when depth limit is reached

        valid_actions = current_env.get_valid_actions()

        if maximizing_player:
            value = -float('inf')
            min_depth_to_terminal_child = float('inf') # Initialize to find minimum depth to terminal among children
            for action in valid_actions:
                next_env = current_env.clone() # Optimized clone
                _ = next_env.step(action)
                state_before_recursion = next_env.board.clone().numpy() # State BEFORE recursive call for data generation
                val, depth_to_terminal_child = self.alpha_beta(next_env, depth - 1, alpha, beta, False, current_depth_in_search + 1) # Increment depth here, get depth_to_terminal
                min_depth_to_terminal_child = min(min_depth_to_terminal_child, depth_to_terminal_child + 1) # Current state's depth is 1 + child's depth
                if val > value:
                    value = val

                if self.data_generation_mode and self.data_buffer is not None and isinstance(self.data_buffer, list) and len(self.data_buffer) >= 2:
                    if len(self.data_buffer) < 3 or (len(self.data_buffer) >=3 and len(self.data_buffer) < (NUM_STATES_PER_OUTCOME * 3 + 3)): # Example max buffer size
                        if depth_to_terminal_child + 1 > MIN_BACKWARD_DEPTH_TO_TERMINAL_SAVE: # Save if depth to terminal is greater than threshold
                            if val == float('inf') and self.data_buffer[0] < NUM_STATES_PER_OUTCOME:
                                self.data_buffer.append((state_before_recursion, 1))
                            elif val == float('-inf') and self.data_buffer[1] < NUM_STATES_PER_OUTCOME:
                                self.data_buffer.append((state_before_recursion, -1))

                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            self.transposition_table[state_tuple] = (value, depth, min_depth_to_terminal_child) # Store in transposition table, store depth_to_terminal
            return value, min_depth_to_terminal_child # Return depth_to_terminal

        else: # Minimizing player (Opponent)
            value = float('inf')
            min_depth_to_terminal_child = float('inf') # Initialize to find minimum depth to terminal among children
            for action in valid_actions:
                next_env = current_env.clone() # Optimized clone
                _ = next_env.step(action)
                state_before_recursion = next_env.board.clone().numpy() # State BEFORE recursive call for data generation
                val, depth_to_terminal_child = self.alpha_beta(next_env, depth - 1, alpha, beta, True, current_depth_in_search + 1) # Increment depth here, get depth_to_terminal
                min_depth_to_terminal_child = min(min_depth_to_terminal_child, depth_to_terminal_child + 1) # Current state's depth is 1 + child's depth
                if val < value:
                    value = val

                if self.data_generation_mode and self.data_buffer is not None and isinstance(self.data_buffer, list) and len(self.data_buffer) >= 2:
                    if len(self.data_buffer) < 3 or (len(self.data_buffer) >=3 and len(self.data_buffer) < (NUM_STATES_PER_OUTCOME * 3 + 3)): # Example max buffer size
                        if depth_to_terminal_child + 1 > MIN_BACKWARD_DEPTH_TO_TERMINAL_SAVE: # Save if depth to terminal is greater than threshold
                            if val == float('inf') and self.data_buffer[0] < NUM_STATES_PER_OUTCOME:
                                self.data_buffer.append((state_before_recursion, -1)) # Opponent win is loss for agent
                            elif val == float('-inf') and self.data_buffer[1] < NUM_STATES_PER_OUTCOME:
                                self.data_buffer.append((state_before_recursion, 1)) # Opponent loss is win for agent

                beta = min(beta, value)
                if beta <= alpha:
                    break
            self.transposition_table[state_tuple] = (value, depth, min_depth_to_terminal_child) # Store in transposition table, store depth_to_terminal
            return value, min_depth_to_terminal_child # Return depth_to_terminal

    def select_action(self, state_tensor):
        valid_actions = self.env.get_valid_actions()
        if not valid_actions:
            return None # No valid actions, game over probably

        action_values = {}
        alpha = -float('inf')
        beta = float('inf')

        for action in valid_actions:
            next_env = self.env.clone() # Optimized clone
            _ = next_env.step(action)
            value, _ = self.alpha_beta(next_env, self.depth - 1, alpha, beta, False) # Get value, but depth_to_terminal is not used here for action selection
            action_values[action] = value

        # Rule 1: If any winning action exists, choose randomly among them.
        winning_actions = [action for action, val in action_values.items() if val == float('inf')]
        if winning_actions:
            return random.choice(winning_actions)

        # Rule 2: Exclude any action that results in an immediate loss.
        non_losing_actions = [action for action, val in action_values.items() if val != float('-inf')]
        if non_losing_actions:
            return random.choice(non_losing_actions)
        else:
            # Rule 3: If all moves are losing, choose randomly from all valid actions.
            return random.choice(valid_actions)

# --- Reward Network Class Definition ---
class RewardNet(nn.Module):
    def __init__(self):
        super(RewardNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 6 * 7, 256)  # Flatten conv output
        self.fc_reward = nn.Sequential(
            nn.Linear(256, 32),
            nn.Linear(32, 1)
        )

    def forward(self, state):
        state = state.long()
        state_one_hot = F.one_hot(state.to(torch.int64), num_classes=3).float()
        state_reshaped = state_one_hot.view(-1, 3, 6, 7)  # Reshape for CNN
        x = F.relu(self.conv1(state_reshaped))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten conv output
        x = F.relu(self.fc1(x))
        reward = self.fc_reward(x)
        return reward

# --- Custom Dataset and Training Function ---
class ConnectFourDataset(Dataset):
    def __init__(self, data_pairs):
        self.data_pairs = data_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        state, reward = self.data_pairs[idx]
        state_tensor = torch.tensor(state).float()
        reward_tensor = torch.tensor([reward]).float()
        return state_tensor, reward_tensor

def train_reward_network(dataset_path, model_save_path, num_epochs=REWARD_NETWORK_EPOCHS, batch_size=REWARD_NETWORK_BATCH_SIZE,
                         learning_rate=REWARD_NETWORK_LEARNING_RATE, print_frequency=REWARD_NETWORK_PRINT_FREQUENCY):
    torch.serialization.add_safe_globals(['numpy'])
    terminal_state_data = torch.load(dataset_path, weights_only=False)
    dataset = ConnectFourDataset(terminal_state_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    reward_net = RewardNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(reward_net.parameters(), lr=learning_rate)

    for epoch in tqdm(range(num_epochs), desc="Training Reward Network", unit="epoch"):
        running_loss = 0.0
        for i, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False, unit="batch")):
            states, rewards = data
            optimizer.zero_grad()
            reward_predictions = reward_net(states)
            loss = criterion(reward_predictions, rewards)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % print_frequency == print_frequency - 1:
                tqdm.write(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {running_loss / print_frequency:.4f}") # Use tqdm.write for proper tqdm output
                running_loss = 0.0

    print("Finished Training")
    torch.save(reward_net.state_dict(), model_save_path)
    print(f"Trained Reward Network saved to {model_save_path}")

# --- Data Generation Functions ---
def generate_game_data(process_id, data_buffer, lock, target_wins, target_losses):
    env = ConnectFourEnv()
    agent_player1 = AlphaBetaAgent(env, depth=DATA_GENERATION_MAIN_AGENT_DEPTH, player=1, data_generation_mode=True, data_buffer=data_buffer)
    current_opponent_depth = INITIAL_OPPONENT_DEPTH
    agent_player2 = AlphaBetaAgent(env, depth=current_opponent_depth, player=2) # Opponent with initial depth

    main_agent_wins = 0 # Wins since last depth adjustment
    main_agent_losses = 0 # Losses since last depth adjustment

    while True:
        env.reset()
        data_buffer_process = [0, 0] # [wins, losses] for this process, index 0 for wins, 1 for losses
        agent_player1.data_buffer = data_buffer_process # point agent's data buffer to process local buffer
        agent_player2.depth = current_opponent_depth # Ensure opponent depth is updated for each game
        agent_player1.transposition_table.clear() # Clear transposition table per game for main agent. Consider clearing for opponent too if needed for memory.
        agent_player2.transposition_table.clear()

        current_player = 1
        agents = {1: agent_player1, 2: agent_player2}

        while not env.is_terminal():
            agent = agents[current_player]
            action = agent.select_action(env.board)
            if action is None: # Should not happen in Connect Four unless terminal state is missed.
                print(f"Process {process_id}: No action selected, game ending unexpectedly.")
                break
            _, _, _, _ = env.step(action)
            current_player = 3 - current_player

        game_result = env.get_result()

        reward = None # Initialize reward outside lock
        with lock:
            if game_result == 1 and data_buffer[0] < target_wins: # Player 1 win (our agent)
                data_buffer[0] += 1
                main_agent_wins += 1
                reward = 1
            elif game_result == -1 and data_buffer[1] < target_losses: # Player 2 win (opponent)
                data_buffer[1] += 1
                main_agent_losses += 1
                reward = -1
            # else: draw, reward remains None

            if reward is not None and len(data_buffer_process) > 2:
                for state, outcome_value in data_buffer_process[2:]:
                    # Instead of checking for 100, check for the expected values
                    if outcome_value in [1, -1]:
                        if reward == 1 and data_buffer[0] < target_wins:
                            data_buffer.append((state, reward))
                            data_buffer[0] += 1  # Increment win counter
                            # Append symmetrical flipped state
                            flipped_state = flip_board(state)
                            data_buffer.append((flipped_state, reward))
                            data_buffer[0] += 1
                        elif reward == -1 and data_buffer[1] < target_losses:
                            data_buffer.append((state, reward))
                            data_buffer[1] += 1  # Increment loss counter
                            # Append symmetrical flipped state
                            flipped_state = flip_board(state)
                            data_buffer.append((flipped_state, reward))
                            data_buffer[1] += 1

        game_count = main_agent_wins + main_agent_losses

        if not (game_count % OPPONENT_DEPTH_ADJUST_FREQUENCY):
            if main_agent_losses > 0:
                win_loss_ratio = main_agent_wins / main_agent_losses
            else:
                win_loss_ratio = float('inf') # Avoid division by zero if no losses yet.

            if main_agent_wins > 0:
                loss_win_ratio = main_agent_losses / main_agent_wins
            else:
                loss_win_ratio = float('inf') # Avoid division by zero if no wins yet.


            if win_loss_ratio > OPPONENT_DEPTH_ADJUST_RATIO:
                current_opponent_depth = min(MAX_OPPONENT_DEPTH, current_opponent_depth + 1) # Increase depth, but cap at MAX_OPPONENT_DEPTH
                tqdm.write(f"Process {process_id}: Win/Loss ratio exceeded {OPPONENT_DEPTH_ADJUST_RATIO}. Increasing opponent depth to {current_opponent_depth}. Total win: {main_agent_wins}, Total loss: {main_agent_losses} ")
            elif loss_win_ratio > OPPONENT_DEPTH_ADJUST_RATIO:
                current_opponent_depth = max(MIN_OPPONENT_DEPTH, current_opponent_depth - 1) # Decrease depth, but keep at least MIN_OPPONENT_DEPTH
                tqdm.write(f"Process {process_id}: Loss/Win ratio exceeded {OPPONENT_DEPTH_ADJUST_RATIO}. Decreasing opponent depth to {current_opponent_depth}. Total win: {main_agent_wins}, Total loss: {main_agent_losses}")

            agent_player2.depth = current_opponent_depth # Update opponent agent's depth

        # Removed process-specific tqdm updates from here.

        if data_buffer[0] >= target_wins and data_buffer[1] >= target_losses:
            tqdm.write(f"Process {process_id}: Target data reached. Exiting.")
            break

    # Removed progress_bar.close()


def flip_board(board_np):
    flipped_board = board_np.copy()
    for row in range(flipped_board.shape[0]): # Iterate over rows
        flipped_board[row, :] = flipped_board[row, ::-1] # Reverse columns for each row
    return flipped_board

def generate_terminal_states_parallel():
    manager = mp.Manager()
    lock = manager.Lock()
    data_buffer = manager.list([0, 0]) # [win_count, loss_count] - shared list
    num_processes = mp.cpu_count()
    processes = []
    target_wins = NUM_STATES_PER_OUTCOME
    target_losses = NUM_STATES_PER_OUTCOME
    total_target_states = target_wins + target_losses

    print(f"Starting {num_processes} processes for data generation...")
    overall_progress_bar = tqdm(total=total_target_states, desc=f"Total Data Generation (Wins: {data_buffer[0]}, Losses: {data_buffer[1]})", position=0, leave=True) # Single tqdm

    for i in range(num_processes):
        p = mp.Process(target=generate_game_data, args=(i, data_buffer, lock, target_wins, target_losses, ))
        processes.append(p)
        p.start()

    while True:
        time.sleep(1) # Check progress every 1 second
        current_wins = data_buffer[0]
        current_losses = data_buffer[1]
        current_total = current_wins + current_losses
        overall_progress_bar.n = current_total
        overall_progress_bar.set_description(f"Total Data Generation (Wins: {current_wins}, Losses: {current_losses})") # Update description with win/loss counts
        overall_progress_bar.refresh()

        if current_total >= total_target_states:
            break

    for p in processes:
        p.join()

    overall_progress_bar.close()
    print("Data generation processes finished.")
    # Convert managed list to regular list for saving
    return list(data_buffer[2:]) if len(data_buffer) > 2 else [] # return states from index 2 onwards

# --- Main Execution Block ---
if __name__ == '__main__':
    # print("Generating balanced state data (Win/Loss only - excluding terminal states, saving non-terminal states with backward_depth_to_terminal > 2) in parallel with dynamic opponent depth and symmetrical flip and Transposition Table...")
    # combined_data = generate_terminal_states_parallel()

    # Verify actual counts in the combined dataset.
    # win_count = sum(1 for state, reward in combined_data if reward == 1)
    # loss_count = sum(1 for state, reward in combined_data if reward == -1)
    # print(f"Combined dataset balance check: Wins: {win_count}, Losses: {loss_count}")

    # Trim to exactly NUM_STATES_PER_OUTCOME states per outcome. # No trimming needed, generation already limits it.
    # balanced_data = combined_data # Already balanced by generation process
    # print(f"Balanced dataset size: {len(balanced_data)}")

    # torch.save(balanced_data, DATASET_PATH)
    # print(f"Generated balanced states and saved to {DATASET_PATH}")

    print("Starting Reward Network training with balanced data...")
    train_reward_network(DATASET_PATH, MODEL_SAVE_PATH)
    print(f"Reward Network training complete. Model saved to {MODEL_SAVE_PATH}")