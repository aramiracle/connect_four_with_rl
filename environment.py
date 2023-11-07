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

    def reset(self):
        # Reset the board and randomly select which player starts
        self.board = torch.zeros((6, 7), dtype=torch.float32)
        self.current_player = torch.randint(1, 3, ()).item()
        self.winner = None
        # Return initial observation
        return self.board * self.current_player

    def step(self, action):
        # Process the player's action and update the game state
        if self.winner is not None:
            # If the game is over, return the final state
            return self.board, 0, True, {}
        
        # Place the current player's piece in the selected column
        for row in range(5, -1, -1):
            if self.board[row][action] == 0:
                self.board[row][action] = self.current_player
                break
        else:
            # If the column is full, return a penalty
            return self.board, -1, False, {}

        # Check for a win or a tie, and assign rewards accordingly
        if self.check_win(row, action):
            self.winner = self.current_player
            reward = 1.0
            done = True
        elif torch.count_nonzero(self.board) == self.max_moves:
            reward = (self.max_moves - 1) / self.max_moves
            done = True
        else:
            reward = -1 / self.max_moves
            done = False

        # Alternate turns
        self.current_player = 3 - self.current_player
        return self.board, reward, done, {}

    def render(self, mode='human'):
        # Display the board to the console
        print(self.board)

    def check_win(self, row, col):
        # Check if the last move completed a line of four of the same player's pieces
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            # Check in each direction from the last piece placed
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
