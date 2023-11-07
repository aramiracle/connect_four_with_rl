# Connect Four AI with DQN

This project implements a playable Connect Four game with a graphical user interface (GUI), where the opponent is an AI trained via a Deep Q-Network (DQN). The AI learns optimal gameplay by training against itself and saves its learned strategy to compete with human players.

## Scripts Overview

### `main.py`

This script is the entry point of the Connect Four game. It creates a window application using the PyQt6 framework and manages the game's state and player interactions. The game board is represented as a grid of buttons. When a button is clicked, the game logic determines if the current player or AI has won, or if the game is a draw. The script also allows for the loading of a pre-trained DQN model to play against the AI.

Key Features:
- **Window and Board Initialization**: Initializes the game window and board layout.
- **Move Handling**: Handles player and AI moves.
- **State and UI Update**: Updates the game state and UI after each move.
- **DQN Model Loading**: Loads a trained DQN agent to play against.
- **Game Termination Check**: Checks for game termination conditions (win, loss, draw).

### `environment.py`

This script defines the `ConnectFourEnv` class that simulates the Connect Four game environment. It inherits from the `gym.Env` class, making it compatible with the OpenAI Gym interface. The environment tracks the state of the game board and provides a `step` function to make moves, a `reset` function to start a new game, and a `render` function to display the game state in the console.

Key Features:
- **Connect Four Logic Implementation**: Implements the Connect Four rules and game logic.
- **Game Board Management**: Manages the game board state as a 2D tensor.
- **Agent Interaction Interface**: Provides an interface for agents to interact with the environment.
- **Legal Move Determination**: Determines the legal moves and updates the board accordingly.
- **Winning and Draw Check**: Checks for a winning condition or a draw.

### `dqn.py`

This script contains the neural network architecture for the DQN (`DQN` class), the experience replay mechanism (`ExperienceReplayBuffer` class), and the DQN agent (`DQNAgent` class). It details the model's layers and forward pass, the methodology for storing and sampling experiences, and the agent's policy for selecting actions and learning from experiences.

Key Features:
- **DQN Model Architecture**: Defines the DQN model architecture using PyTorch.
- **Experience Replay Management**: Manages the experience replay buffer for efficient learning.
- **Agent Learning Loop**: Implements the agent's learning loop, including action selection and model updates.
- **Training Process Conduct**: Conducts the training process over a series of episodes.
- **Model Saving**: Saves the trained model's parameters for later use.

### `test.py`

The `test.py` script is designed to evaluate the performance of the trained DQN agent. It tests the agent against a random opponent over a number of games, alternating between starting first and second. After the simulation, it reports the number of wins, losses, and draws to assess the AI's skill level.

Key Features:
- **AI Performance Simulation**: Simulates games between the trained DQN agent and a random bot.
- **Statistical Performance Evaluation**: Evaluates the AI's performance statistically over multiple games.
- **Winning Ability Insight**: Provides insights into the AI's ability to win as both the first and second player.
- **Strategy Improvement Assistance**: Helps to debug and improve the agent's strategy after training.

### `upgraded_dqn.py`

This script provides an improved version of the DQN model for the Connect Four game. Key enhancements include:

- **Convolutional Network Architecture**: Uses convolutional layers to process the game state, capturing spatial hierarchies.
- **Dueling Network Architecture**: Implements a dueling architecture that separates the estimation of the value and advantage streams.
- **Priority Experience Replay Enhancement**: Enhances the replay mechanism to prioritize important experiences, accelerating learning.
- **Epsilon-Greedy Action Selection**: Employs an epsilon-greedy approach for action selection, balancing exploration and exploitation.
- **Comprehensive Training Loop**: Includes a comprehensive training loop with target network updates for stability.

### `mct.py`

This script introduces a Monte Carlo Tree Search (MCTS) agent for strategic gameplay:

- **Monte Carlo Tree Search Algorithm**: Implements MCTS, a search algorithm for making decisions in a game tree without requiring high computational costs.
- **Node Expansion Strategy**: Expands the search tree based on random simulations and updates nodes with the outcomes.
- **Exploration vs. Exploitation Balance**: Uses the Upper Confidence Bound (UCB1) formula to balance between exploring new moves and exploiting known good ones.
- **Value Backpropagation**: After each simulation, updates the node values and visit counts up the tree.

### `gui.py`

This script uses the PyQt6 framework to create a GUI for the Connect Four game:

- **PyQt6 User Interface Creation**: Leverages PyQt6 to create a responsive and visually appealing user interface.
- **Interactive Game Board**: Presents a 6x7 grid of buttons that players can click to make a move.
- **Game Logic Handling**: Includes methods to handle the game logic, such as player moves, win checks, and draw conditions.
- **Game State Status Updates**: Provides a status label on the GUI to inform players about the current game state or endgame results.

Each script collaborates to create a complete Connect Four experience with a challenging AI opponent. Users can interact with the game, train the AI, and evaluate its gameplay, all through these Python scripts.

## Installation

```
pip install PyQt6 torch gymnasium numpy tqdm
```

## Usage

