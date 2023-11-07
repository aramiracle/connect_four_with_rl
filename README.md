# Connect Four AI with DQN

This project implements a playable Connect Four game with a graphical user interface (GUI), where the opponent is an AI trained via a Deep Q-Network (DQN). The AI learns optimal gameplay by training against itself and saves its learned strategy to compete with human players.

## Scripts Overview

### `main.py`

This script is the entry point of the Connect Four game. It creates a window application using the PyQt6 framework and manages the game's state and player interactions. The game board is represented as a grid of buttons. When a button is clicked, the game logic determines if the current player or AI has won or if the game is a draw. The script also allows loading a pre-trained DQN model to play against the AI.

Key Features:
- Initializes the game window and board layout
- Handles player moves and AI moves
- Updates the game state and UI after each move
- Loads a trained DQN agent to play against
- Checks for game termination conditions (win, lose, draw)

### `environment.py`

This script defines the `ConnectFourEnv` class that simulates the Connect Four game environment. It inherits from the `gym.Env` class, making it compatible with the OpenAI Gym interface. The environment tracks the state of the game board and provides a `step` function to make moves, a `reset` function to start a new game, and a `render` function to display the game state in the console.

Key Features:
- Implements the Connect Four rules and game logic
- Manages the game board state as a 2D tensor
- Provides an interface for agents to interact with the environment
- Determines the legal moves and updates the board accordingly
- Checks for a winning condition or a draw

### `dqn.py`

This script contains the neural network architecture for the DQN (`DQN` class), the experience replay mechanism (`ExperienceReplayBuffer` class), and the DQN agent (`DQNAgent` class). It details the model's layers and forward pass, the methodology for storing and sampling experiences, and the agent's policy for selecting actions and learning from experiences.

Key Features:
- Defines the DQN model architecture using PyTorch
- Manages the experience replay buffer for efficient learning
- Implements the agent's learning loop, including action selection and model updates
- Conducts the training process over a series of episodes
- Saves the trained model's parameters for later use

### `test.py`

The `test.py` script is designed to evaluate the performance of the trained DQN agent. It tests the agent against a random opponent over a number of games, alternating between starting first and second. After the simulation, it reports the number of wins, losses, and draws to assess the AI's skill level.

Key Features:
- Simulates games between the trained DQN agent and a random bot
- Evaluates the AI's performance statistically over multiple games
- Provides insights into the AI's ability to win as both the first and second player
- Helps to debug and improve the agent's strategy after training

Each script collaborates to create a complete Connect Four experience with a challenging AI opponent. Users can interact with the game, train the AI, and evaluate its gameplay, all through these Python scripts.
