import gymnasium as gym
import numpy as np
import torch

# Custom environment for Connect Four game following the gym interface
class ConnectFourEnv(gym.Env):
    def __init__(self):
        # Initialize the Connect Four board as a 6x7 grid
        self.board = torch.zeros((6, 7), dtype=torch.float32)
        # Player 1 starts
        self.current_player = 1
        self.winner = None
        # Connect Four ends after 42 moves (a full board)
        self.max_moves = 42
        # Define the action and observation space according to gym's API
        self.action_space = gym.spaces.Discrete(7)
        self.observation_space = gym.spaces.Box(low=0, high=2, shape=(6, 7), dtype=np.float32)
        # Variables to track the last move
        self.last_row = None
        self.last_col = None

    def reset(self, initial_board=None):
        if initial_board is not None:
            # Reset the board to the provided initial_board
            self.board = torch.tensor(initial_board, dtype=torch.float32).clone() # Ensure it's a tensor and cloned
        else:
            # Default reset: empty board
            self.board = torch.zeros((6, 7), dtype=torch.float32)
        self.current_player = 1 # Can be randomized if needed: 1 if np.random.rand() < 0.5 else 2
        self.winner = None
        # Reset last move
        self.last_row = None
        self.last_col = None
        # Return initial observation
        return self.board # Gym v26 requires returning observation and info

    def step(self, action):
        # Ensure action is valid (within action space and column is not full)
        if not 0 <= action < 7:
            raise ValueError(f"Invalid action {action}")
        if self.board[0][action] != 0:
            # Invalid move (column full), return a large negative reward and terminate? Or just invalid action
            # For now, let's assume environment handles only valid actions from agent.
            pass # In a real scenario, handle invalid action more explicitly

        # Place the current player's piece in the selected column
        for row in range(5, -1, -1):
            if self.board[row][action] == 0:
                self.board[row][action] = self.current_player
                self.last_row = row
                self.last_col = action
                break

        reward = self.calculate_reward(row, action) # Calculate reward based on the move

        terminated = False
        truncated = False # For time limits, not used in Connect Four usually

        if  self.check_win(row, action):
            self.winner = self.current_player
            info = {'winner': f'Player {self.current_player}'}
            reward += 1000 # Big win reward on top of alignment and strategic rewards
            terminated = True
        elif torch.count_nonzero(self.board) == self.max_moves:
            reward += 500 # Draw reward
            info = {'winner': 'Draw'}
            terminated = True
        else: # Game is not terminal
            info = {'winner': 'Game is not finished yet.'}
            reward -= 1 # Small step penalty to encourage faster games
            terminated = False

        # Alternate turns
        self.current_player = 3 - self.current_player
        return self.board, reward, terminated, info # Gym v26 requires terminated, truncated, info


    def calculate_reward(self, row, col):
        reward = 0

        # 1. Reward for Alignments (starts rewarding from 2 in a row)
        alignment_reward = 0
        current_player = self.board[row][col]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            alignment_count = self.count_aligned(row, col, dr, dc, current_player)
            if alignment_count >= 2: # Start rewarding from 2 in a row
                alignment_reward += 10**(alignment_count - 2) # Reduced base and exponent

        reward += alignment_reward

        # 2. Blocking Opponent's Winning Move (Positive Reward)
        if self.check_block_opponent_win(row, col):
            reward += 150  # Reward for blocking a direct win for opponent

        # 3. Creating a Threat (Creating a position where you can win on the next turn - Positive Reward)
        if self.check_create_winning_threat(row, col):
            reward += 100 # Reward for creating a winning threat

        # 4. Penalize Giving Opponent a Winning Move (Negative Reward)
        if self.check_give_opponent_winning_move(row, col):
            reward -= 250 # Penalty for making a move that allows opponent to win immediately

        return reward


    def render(self):
        print(self.board)

    def is_terminal(self):
        # Check if there's a win or a tie
        for row in range(6):
            for col in range(7):
                if self.board[row, col] != 0 and self.check_win(row, col):
                    return True
        return torch.count_nonzero(self.board) == self.max_moves

    def check_win(self, row, col):
        # Check if there's a winner starting from the given position (row, col)
        if self.board[row, col] == 0: # No piece at this position, cannot be a win
            return False
        current_player = self.board[row][col]
        # Check all directions for a win condition
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)] # Down, Right, Diagonal (down-right), Diagonal (down-left)
        for dr, dc in directions:
            if self.count_aligned(row, col, dr, dc, current_player) >= 4:
                return True
        return False

    def count_aligned(self, row, col, dr, dc, player):
        # Count consecutive pieces of 'player' starting from (row, col) in direction (dr, dc)
        count = 1 # Count the piece at (row, col) itself
        count += self.count_direction(row, col, dr, dc, 1, player) # Count in positive direction
        count += self.count_direction(row, col, dr, dc, -1, player) # Count in negative direction
        return count

    def count_direction(self, row, col, dr, dc, step, player):
        # Helper function for count_aligned to count in one direction
        count = 0
        for i in range(1, 4): # Check up to 3 more pieces in the direction
            r, c = row + dr * i * step, col + dc * i * step
            if 0 <= r < 6 and 0 <= c < 7 and self.board[r][c] == player:
                count += 1
            else:
                break # Stop if out of bounds or not the same player
        return count

    def get_result(self):
        # If the game is terminal, determine if it is a win, loss, or draw for the current player
        if not self.is_terminal():
            return None  # Game is still ongoing

        if self.winner == self.current_player:
            return 1  # Current player won
        elif self.winner is not None: # There is a winner, but it's not the current player, so current player lost.
            return -1  # Current player lost
        else:
            return 0  # Draw

    def get_valid_actions(self):
        # Returns a list of valid column indices where a piece can be placed
        return [col for col in range(self.board.shape[1]) if self.board[0, col] == 0]

    def get_last_move(self):
        # Returns the last move made (row, col)
        return self.last_row, self.last_col

    def clone(self):
        # Returns a deep copy of the environment
        new_env = ConnectFourEnv()
        new_env.board = self.board.clone()
        new_env.current_player = self.current_player
        new_env.winner = self.winner
        new_env.last_row = self.last_row
        new_env.last_col = self.last_col
        return new_env


    # --- Helper Functions for Enhanced Rewards ---

    def check_block_opponent_win(self, last_row, last_col):
        """Reward for blocking an immediate winning move of the opponent."""
        opponent_player = 3 - self.current_player
        env_copy = self.clone()
        env_copy.current_player = opponent_player # Set to opponent's turn for simulation

        # Revert the last move to simulate *before* current player's move, to check what the opponent could have done.
        env_copy.board[last_row][last_col] = 0

        for col in range(env_copy.board.shape[1]):
            if env_copy.board[0][col] == 0: # Valid move column for opponent
                for row in range(5, -1, -1):
                    if env_copy.board[row][col] == 0:
                        env_copy.board[row][col] = opponent_player # Opponent makes a potential move
                        if env_copy.check_win(row, col): # Check if it's a winning move for opponent
                            return True # Current player's move blocked opponent's win
                        env_copy.board[row][col] = 0 # Reset for next check
                        break
        return False


    def check_create_winning_threat(self, last_row, last_col):
        """Reward for creating a position where the current player can win on the next turn."""
        current_player = self.board[last_row][last_col]
        env_copy = self.clone()
        env_copy.current_player = self.current_player # Ensure it's current player's turn in copy

        for col in range(env_copy.board.shape[1]):
            if env_copy.board[0][col] == 0: # Valid move column for current player
                for row in range(5, -1, -1):
                    if env_copy.board[row][col] == 0:
                        env_copy.board[row][col] = current_player # Simulate next move by current player
                        if env_copy.check_win(row, col): # Check if this move is a winning move
                            return True # Current move created a winning threat for next turn
                        env_copy.board[row][col] = 0 # Reset
                        break
        return False

    def check_give_opponent_winning_move(self, last_row, last_col):
        """Penalize making a move that allows the opponent to win on their next turn."""
        opponent_player = 3 - self.current_player
        env_copy = self.clone()
        env_copy.current_player = opponent_player # Set to opponent's turn for simulation

        # Check from the perspective of the *opponent* after the current player's move.
        for col in range(env_copy.board.shape[1]):
            if env_copy.board[0][col] == 0: # Valid move column for opponent
                for row in range(5, -1, -1):
                    if env_copy.board[row][col] == 0:
                        env_copy.board[row][col] = opponent_player # Opponent makes a potential move
                        if env_copy.check_win(row, col): # Check if it's a winning move for opponent
                            return True # Current move allowed opponent to win on their turn
                        env_copy.board[row][col] = 0 # Reset for next check
                        break
        return False