import gymnasium as gym
import numpy as np
import torch

# Custom environment for Connect Four game following the gym interface
class ConnectFourEnv(gym.Env):
    def __init__(self):
        # Initialize the Connect Four board as a 6x7 grid
        self.board = torch.zeros((6, 7), dtype=torch.float32)
        # Start with player 1
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

    def reset(self):
        # Reset the board and randomly select which player starts
        self.board = torch.zeros((6, 7), dtype=torch.float32)
        self.current_player = 1
        self.winner = None
        # Reset last move
        self.last_row = None
        self.last_col = None
        # Return initial observation
        return self.board

    def step(self, action):
        # Place the current player's piece in the selected column
        for row in range(5, -1, -1):
            if self.board[row][action] == 0:
                self.board[row][action] = self.current_player
                # Update last move
                self.last_row = row
                self.last_col = action
                break

        # Check for an instant loss and assign rewards accordingly
        if  self.check_win(row, action):
            self.winner = self.current_player
            info = {'winner': f'Player {self.current_player}'}
            reward = 100.0
            done = True
        elif torch.count_nonzero(self.board) == self.max_moves:
            reward = 50
            info = {'winner': 'Draw'}
            done = True
        elif self.check_instant_loss(row, action):
            info = {'winner': 'Game is not finished yet.'}
            reward = -10.0
            done = False
        else:
            reward = -1
            info = {'winner': 'Game is not finished yet.'}
            done = False

        # Alternate turns
        self.current_player = 3 - self.current_player
        return self.board, reward, done, info

    # Add a method to check for instant loss
    def check_instant_loss(self, last_row, last_col):
        current_player = 3 - self.current_player  # Opponent's player
        env_copy = self.clone()
        # Check if the opponent has a winning move
        for col in range(env_copy.board.shape[1]):
            if env_copy.board[0][col] == 0:
                # Simulate the opponent's move
                for row in range(5, -1, -1):
                    if env_copy.board[row][col] == 0:
                        env_copy.board[row][col] = current_player
                        if env_copy.check_win(row, col):
                            # Reset the board to its original state
                            env_copy.board[last_row][last_col] = 0
                            return True
                        break
        return False


    def render(self):
        # Display the board to the console
        print(self.board)

    def is_terminal(self):
        # Check if there's a win or a tie
        for row in range(6):
            for col in range(7):
                if self.board[row, col] != 0 and self.check_win(row, col):
                    return True
        return torch.count_nonzero(self.board) == self.max_moves

    # Check if there's a winner
    def check_win(self, row, col):
        current_player = self.board[row][col]
        # Check all directions for a win condition
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            if self.count_aligned(row, col, dr, dc, current_player) >= 4:
                return True
        return False

    # Count consecutive pieces in a direction
    def count_aligned(self, row, col, dr, dc, player):
        count = 1
        count += self.count_direction(row, col, dr, dc, 1, player)
        count += self.count_direction(row, col, dr, dc, -1, player)
        return count

    # Count in a single direction
    def count_direction(self, row, col, dr, dc, step, player):
        count = 0
        for i in range(1, 4):
            r, c = row + dr * i * step, col + dc * i * step
            if 0 <= r < 6 and 0 <= c < 7 and self.board[r][c] == player:
                count += 1
            else:
                break
        return count
    
    def get_result(self):
        # If the game is terminal, determine if it is a win, loss, or draw
        if self.is_terminal():
            if self.winner == self.current_player:
                return 1  # Current player won
            elif self.winner is not None:
                return -1  # Current player lost
            else:
                return 0  # Draw
        else:
            return None  # Game is still ongoing
    
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
