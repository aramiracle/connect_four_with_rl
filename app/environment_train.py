import gym
import numpy as np
import torch
from copy import deepcopy

# Optimized Custom environment for Connect Four game following the gym interface
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

    def reset(self, initial_board=None, seed=None, options=None):
        super().reset(seed=seed)
        if initial_board is not None:
            self.board = torch.tensor(initial_board, dtype=torch.float32).clone()
        else:
            self.board = torch.zeros((6, 7), dtype=torch.float32)
        self.current_player = 1
        self.winner = None
        self.last_row = None
        self.last_col = None
        return self.board

    def step(self, action):
        if not 0 <= action < 7:
            raise ValueError(f"Invalid action {action}")

        for row in range(5, -1, -1):
            if self.board[row][action] == 0:
                self.board[row][action] = self.current_player
                self.last_row = row
                self.last_col = action
                break

        reward = self.calculate_reward(row, action, last_action=action)

        terminated = False
        info = {}

        if self.check_win(row, action):
            self.winner = self.current_player
            info = {'winner': f'Player {self.current_player}'}
            reward += 1
            terminated = True
        elif torch.count_nonzero(self.board) == self.max_moves:
            info = {'winner': 'Draw'}
            reward += 0.5
            terminated = True
        else:
            info = {'winner': 'Game is not finished yet.'}
            reward -= 0.05
            terminated = False

        self.current_player = 3 - self.current_player

        return self.board, reward, terminated, info # Return truncated as False

    def calculate_reward(self, row, col, last_action):
        reward = 0

        alignment_reward = 0
        current_player = self.board[row][col]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            alignment_count = self.count_aligned(row, col, dr, dc, current_player)
            if alignment_count >= 2:
                alignment_reward += 0.05 * (alignment_count - 1)

        reward += alignment_reward

        if self.check_block_opponent_win(row, col):
            reward += 0.3

        if self.check_create_winning_threat(row, col):
            reward += 0.2

        if self.check_give_opponent_winning_move(row, col):
            reward -= 0.7

        if self.check_missed_win(last_action):
            reward -= 0.8

        if self.check_forced_win(last_action): # Use last_action here, not action (which is col)
            reward += 0.8

        return reward

    def render(self):
        print(self.board)

    def is_terminal(self):
        for row in range(6):
            for col in range(7):
                if self.board[row, col] != 0 and self.check_win(row, col):
                    return True
        return torch.count_nonzero(self.board) == self.max_moves

    def check_win(self, row, col):
        if self.board[row, col] == 0:
            return False
        player = self.board[row, col]
        # Directions to check: horizontal, vertical, diagonal (up-right), diagonal (up-left)
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)] # Right, Down, Down-Right, Down-Left

        for dr, dc in directions:
            count = 0
            for i in range(4): # Check 4 positions in each direction
                r, c = row + i * dr, col + i * dc
                if 0 <= r < 6 and 0 <= c < 7 and self.board[r, c] == player:
                    count += 1
                else:
                    break # Stop if out of bounds or not player's piece
            if count == 4:
                return True
        return False

    def count_aligned(self, row, col, dr, dc, player):
        count = 0
        opponent = 3 - player

        # Positive direction
        skipped_zero_pos = False
        for i in range(1, 4):
            r, c = row + dr * i, col + dc * i
            if not (0 <= r < 6 and 0 <= c < 7):
                break
            cell = self.board[r, c]
            if cell == player:
                count += 1
            elif cell == 0 and not skipped_zero_pos:
                skipped_zero_pos = True
                continue # Skip one zero
            else:
                break

        # Negative direction
        skipped_zero_neg = False
        for i in range(1, 4):
            r, c = row - dr * i, col - dc * i
            if not (0 <= r < 6 and 0 <= c < 7):
                break
            cell = self.board[r, c]
            if cell == player:
                count += 1
            elif cell == 0 and not skipped_zero_neg:
                skipped_zero_neg = True
                continue # Skip one zero
            else:
                break
        return count + 1


    def get_result(self):
        if not self.is_terminal():
            return None
        if self.winner == self.current_player:
            return 1
        elif self.winner is not None:
            return -1
        else:
            return 0

    def get_valid_actions(self):
        return [col for col in range(self.board.shape[1]) if self.board[0, col] == 0]

    def get_last_move(self):
        return self.last_row, self.last_col

    def clone(self):
        new_env = ConnectFourEnv()
        new_env.board = self.board.clone()
        new_env.current_player = self.current_player
        new_env.winner = self.winner
        new_env.last_row = self.last_row
        new_env.last_col = self.last_col
        return new_env

    # --- Helper Functions for Enhanced Rewards ---

    def check_block_opponent_win(self, last_row, last_col):
        cloned_env = self.clone()
        opponent_player = 3 - self.current_player

        for col in range(cloned_env.board.shape[1]):
            if cloned_env.board[0][col] == 0:
                for row in range(5, -1, -1):
                    if cloned_env.board[row][col] == 0:
                        cloned_env.board[row][col] = opponent_player
                        if cloned_env.check_win(row, col):
                            return True
                        cloned_env.board[row][col] = 0
                        break
        return False


    def check_create_winning_threat(self, last_row, last_col):
        cloned_env = self.clone()
        current_player = self.board[last_row][last_col]

        for col in range(cloned_env.board.shape[1]):
            if cloned_env.board[0][col] == 0:
                for row in range(5, -1, -1):
                    if cloned_env.board[row][col] == 0:
                        cloned_env.board[row][col] = current_player
                        if cloned_env.check_win(row, col):
                            return True
                        cloned_env.board[row][col] = 0
                        break
        return False

    def check_give_opponent_winning_move(self, last_row, last_col):
        cloned_env = self.clone()
        opponent_player = 3 - self.current_player

        for col in range(cloned_env.board.shape[1]):
            if cloned_env.board[0][col] == 0:
                for row in range(5, -1, -1):
                    if cloned_env.board[row][col] == 0:
                        cloned_env.board[row][col] = opponent_player
                        if cloned_env.check_win(row, col):
                            return True
                        cloned_env.board[row][col] = 0
                        break
        return False

    def check_missed_win(self, last_action):
        cloned_env = self.clone()
        for col in range(cloned_env.board.shape[1]):
            if cloned_env.board[0][col] == 0:
                for row in range(5, -1, -1):
                    if cloned_env.board[row][col] == 0:
                        cloned_env.board[row][col] = self.current_player
                        if cloned_env.check_win(row, col):
                            cloned_env.board[row][col] = 0
                            if col != last_action:
                                return True
                            else:
                                return False # Action taken was winning
                        cloned_env.board[row][col] = 0
                        break
        return False

    def check_forced_win(self, action): # action here refers to last_action from step
        cloned_env = self.clone()
        forced_win_row = cloned_env._find_forced_win_row(action)
        if forced_win_row == -1:
            return False

        cloned_env.board[forced_win_row, action] = self.current_player

        forced_win_possible = cloned_env._is_forced_win_possible(action)

        return forced_win_possible

    def _find_forced_win_row(self, action):
        """Helper function to find the row for a forced win move. Operates on cloned env."""
        for row in range(5, -1, -1):
            if self.board[row][action] == 0: # Use self.board here to check valid row in cloned env context
                return row
        return -1 # Column is full

    def _is_forced_win_possible(self, forced_action):
        """Helper function to check if a forced win is possible after the forced action. Operates on cloned env."""
        cloned_env = self.clone() # Clone again to simulate opponent moves without affecting outer cloned_env
        current_player = self.current_player
        opponent_player = 3 - current_player
        forced_win_possible = True

        for opponent_col in range(cloned_env.board.shape[1]):
            if cloned_env.board[0][opponent_col] == 0:
                if not cloned_env._has_winning_response_to_opponent_move(opponent_col, current_player, opponent_player):
                    forced_win_possible = False
                    break  # No need to check other opponent moves
        return forced_win_possible

    def _has_winning_response_to_opponent_move(self, opponent_col, current_player, opponent_player):
        """Helper function to check if current player has a winning response to a specific opponent move. Operates on cloned env."""
        cloned_env = self.clone() # Clone again for opponent move simulation
        opponent_move_row = -1
        for row in range(5, -1, -1):
            if cloned_env.board[row][opponent_col] == 0:
                opponent_move_row = row
                break

        if opponent_move_row != -1: # Valid opponent move
            cloned_env.board[opponent_move_row, opponent_col] = opponent_player  # Opponent moves

            has_winning_response = False
            for response_col in range(cloned_env.board.shape[1]):
                if cloned_env.board[0][response_col] == 0:
                    if cloned_env._is_instant_win_move(response_col, current_player):
                        has_winning_response = True
                        break  # Found instant win response
            return has_winning_response
        return False # Invalid opponent move


    def _is_instant_win_move(self, action, player):
        """Helper function to check if a move in a given column is an instant win for the player. Operates on cloned env."""
        cloned_env = self.clone() # Clone for instant win check
        for row in range(5, -1, -1):
            if cloned_env.board[row][action] == 0:
                cloned_env.board[row][action] = player
                if cloned_env.check_win(row, action):
                    return True  # Found an instant win move
                cloned_env.board[row][action] = 0 # Revert in cloned env, though not strictly necessary as it's cloned
                break  # Column checked
        return False # No instant win move in this column