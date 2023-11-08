import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QGridLayout, QWidget, QRadioButton, QHBoxLayout
from hybrid import HybridAgent
from environment import ConnectFourEnv
from functools import partial
from PyQt6.QtCore import Qt
import torch
import copy

class ConnectFour(QMainWindow):
    # Initialization of the Connect Four game window
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Connect Four")
        self.setGeometry(100, 100, 600, 600)
        self.initUI()
        self.agent = HybridAgent(ConnectFourEnv())  # DQN Agent for playing the game
        self.current_player = 1
        self.game_over = False
        self.game_state_history = []

    # Setting up the user interface components
    def initUI(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.grid = QGridLayout()
        self.central_widget.setLayout(self.grid)
        self.board = [[0] * 7 for _ in range(6)]

        # Create grid of buttons for the game board
        for row in range(6):
            for col in range(7):
                button = QPushButton()
                button.setFixedSize(100, 100)
                # Connect the button click to on_click with only the column index
                button.clicked.connect(partial(self.on_click, col))
                self.grid.addWidget(button, row, col)
                self.board[row][col] = button

        # Status label for game messages
        self.status_label = QLabel()
        self.grid.addWidget(self.status_label, 6, 0, 1, 7, alignment=Qt.AlignmentFlag.AlignCenter)

        # Player selection and game control buttons
        button_row_layout = QHBoxLayout()
        self.player1_button = QRadioButton("Start as Player 1")
        self.player2_button = QRadioButton("Start as Player 2")
        self.player1_button.toggled.connect(self.select_player)
        self.player2_button.toggled.connect(self.select_player)
        button_row_layout.addWidget(self.player1_button)
        button_row_layout.addWidget(self.player2_button)
        self.load_button = QPushButton("Load Agent")
        self.load_button.clicked.connect(self.load_agent)
        button_row_layout.addWidget(self.load_button)
        self.play_button = QPushButton("Play Game")
        self.play_button.clicked.connect(self.start_game)
        button_row_layout.addWidget(self.play_button)
        self.play_button.setDisabled(True)
        self.grid.addLayout(button_row_layout, 7, 0, 1, 7)

    # Handling the player selection
    def select_player(self):
        self.current_player = 1 if self.player1_button.isChecked() else 2
        self.play_button.setDisabled(False)

    # Start a new game and initialize the board
    def start_game(self):
        self.play_button.setDisabled(True)
        self.game_over = False
        self.status_label.setText("")
        self.game_state_history = []
        for row in range(6):
            for col in range(7):
                self.board[row][col].setStyleSheet("")
        self.current_player = 1 if self.player1_button.isChecked() else 2
        if self.current_player == 2:
            self.play_ai_turn()

    # Update the UI to reflect the current game state
    def update_board_from_game_state(self, game_state):
        for row in range(6):
            for col in range(7):
                button = self.board[row][col]
                color = 'red' if game_state[row][col] == 1 else 'yellow' if game_state[row][col] == 2 else ''
                button.setStyleSheet(f'background-color: {color};')

    # Handle button click and game logic
    def on_click(self, col):
        # If the game is over or the column is full, ignore the click
        if self.game_over or not self.column_not_full(col):
            return
        
        # Find the first empty space in the column and update the environment
        row = self.update_env(col, self.current_player)
        
        # Now, update only the affected button in the GUI
        self.update_button(row, col, self.current_player)
        
        # Check for win or continue the game
        game_end = self.check_game_end(row, col)
        if not game_end:
            # Switch the current player
            self.current_player = 3 - self.current_player
            self.play_ai_turn()


    def play_ai_turn(self):
        if self.game_over:
            return

        # Create a copy of the environment for MCTS to simulate
        env_copy = copy.deepcopy(self.agent.env)
        
        # Use MCTS to choose an action using the copy of the environment
        action = self.agent.select_action(env_copy, use_mcts=True)
        
        # Apply the chosen action to the actual environment
        row = self.update_env(action, self.current_player)

        # Now update the GUI for the corresponding button
        self.update_button(row, action, self.current_player)

        # Check if AI's action ends the game
        if self.check_game_end(row, action):
            # If the game ended, determine and display the winner
            winner = "AI" if self.agent.env.winner == self.current_player else "Player"
            self.status_label.setText(f"{winner} wins!")
        else:
            # Switch the current player
            self.current_player = 3 - self.current_player


    def update_env(self, col, player):
        for r in range(5, -1, -1):
            if self.agent.env.board[r][col] == 0:
                self.agent.env.board[r][col] = player
                return r
        return None  # If the column is full, this shouldn't happen due to the earlier check

    def update_button(self, row, col, player):
        color = 'red' if player == 1 else 'yellow'
        if row is not None:
            self.board[row][col].setStyleSheet(f'background-color: {color};')

    def check_game_end(self, row, col):
        if row is None:
            return True
        if self.agent.env.check_win(row, col):
            winner = 'Player' if self.current_player == 1 else 'AI'
            self.status_label.setText(f"{winner} wins!")
            self.game_over = True
            return True
        if self.agent.env.is_terminal():
            self.status_label.setText("It's a draw!")
            self.game_over = True
            return True
        return False

    # Check if the column is not full
    def column_not_full(self, column):
        return self.board[0][column].styleSheet() == ""
            
    # Get the current game state
    def get_game_state(self):
        return [[1 if cell.styleSheet() == 'background-color: red;' else 
                 2 if cell.styleSheet() == 'background-color: yellow;' else 0 
                 for cell in row] for row in self.board]

    # Check if there's a winner
    def check_win(self, row, col):
        # Check all directions for a win condition
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            if self.count_aligned(row, col, dr, dc) >= 4:
                return True
        return False

    # Count consecutive pieces in a direction
    def count_aligned(self, row, col, dr, dc):
        count = 1
        count += self.count_direction(row, col, dr, dc, 1)
        count += self.count_direction(row, col, dr, dc, -1)
        return count

    # Count in a single direction
    def count_direction(self, row, col, dr, dc, step):
        count = 0
        for i in range(1, 4):
            r, c = row + dr * i * step, col + dc * i * step
            if 0 <= r < 6 and 0 <= c < 7 and self.board[r][c].styleSheet() == self.board[row][col].styleSheet():
                count += 1
            else:
                break
        return count
    
    # Check for a draw condition
    def check_draw(self):
        return all(self.board[0][col].styleSheet() != "" for col in range(7))

    # Load the trained DQN agent
    def load_agent(self):
        try:
            checkpoint = torch.load('saved_agents/dqn_agent_after_training.pth')
            self.agent.dqn_agent.model.load_state_dict(checkpoint['model_state_dict'])
            self.agent.dqn_agent.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.agent.dqn_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.status_label.setText("Agent loaded successfully.")
            self.play_button.setDisabled(False)
        except FileNotFoundError:
            self.status_label.setText("Agent file not found.")
        except Exception as e:
            self.status_label.setText(f"Failed to load agent: {str(e)}")

# Entry point of the application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ConnectFour()
    window.show()
    sys.exit(app.exec())
