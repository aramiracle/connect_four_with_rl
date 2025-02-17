import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QGridLayout, QWidget, QRadioButton, QHBoxLayout, QInputDialog
from PyQt6.QtCore import Qt
from functools import partial
import torch

from app.hybrid.hybrid import HybridAgent
from app.dqn.dqn import DQNAgent
from app.ddqn.ddqn import DDQNAgent
from app.a3c.a3c import A3CAgent
from app.c51.c51 import C51Agent
from app.ppo.ppo import PPOAgent
from app.sac.sac import SACAgent # Import SACAgent
from app.environment_test import ConnectFourEnv

# Base class for Connect Four Game
class ConnectFour(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Connect Four")
        self.setGeometry(100, 100, 600, 600)
        self.agent = None
        self.current_player = 1
        self.game_over = False
        self.game_state_history = []
        self.initUI()

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
                button.clicked.connect(partial(self.on_click, col))
                button.setEnabled(False)
                self.grid.addWidget(button, row, col)
                self.board[row][col] = button

        # Status label for game messages
        self.status_label = QLabel()
        self.grid.addWidget(self.status_label, 6, 0, 1, 7, alignment=Qt.AlignmentFlag.AlignCenter)

        # Player selection, game control buttons, and agent selection
        button_row_layout = QHBoxLayout()
        self.player1_button = QRadioButton("Start as Player 1")
        self.player2_button = QRadioButton("Start as Player 2")
        self.player1_button.toggled.connect(self.select_player)
        self.player2_button.toggled.connect(self.select_player)
        button_row_layout.addWidget(self.player1_button)
        button_row_layout.addWidget(self.player2_button)
        self.agent_selection_button = QPushButton("Select Agent")
        self.agent_selection_button.clicked.connect(self.select_agent)
        button_row_layout.addWidget(self.agent_selection_button)
        self.play_button = QPushButton("Play Game")
        self.play_button.clicked.connect(self.start_game)
        button_row_layout.addWidget(self.play_button)
        self.play_button.setDisabled(True)
        self.grid.addLayout(button_row_layout, 7, 0, 1, 7)
        self.play_again_button = QPushButton("Play Again")
        self.play_again_button.clicked.connect(self.play_again)
        self.play_again_button.setDisabled(True)
        button_row_layout.addWidget(self.play_again_button)

    def play_again(self):
        self.game_over = False
        self.status_label.setText("")
        self.game_state_history = []

        # Reset the board
        for row in range(6):
            for col in range(7):
                self.board[row][col].setStyleSheet("")
                self.board[row][col].setEnabled(False)  # Disable buttons initially

        # Reset the environment
        if self.agent:
            self.agent.env.reset()

        # Enable the board buttons at the start of the game
        self.enable_board_buttons()

        # Reset the current player based on the radio button selection
        self.current_player = 1 if self.player1_button.isChecked() else 2

        if self.current_player == 2 and self.agent:
            self.play_ai_turn()

    def select_player(self):
        self.current_player = 1 if self.player1_button.isChecked() else 2

    def enable_board_buttons(self):
        for row in range(6):
            for col in range(7):
                self.board[row][col].setEnabled(True)

    def start_game(self):
        self.game_over = False
        self.status_label.setText("")
        self.game_state_history = []
        for row in range(6):
            for col in range(7):
                self.board[row][col].setStyleSheet("")

        # Enable the board buttons at the start of the game
        self.enable_board_buttons()

        self.current_player = 1
        if self.player2_button.isChecked() and self.agent:
            # Start AI's turn for Player 2
            self.play_ai_turn()

    def on_click(self, col):
        if self.game_over or not self.column_not_full(col):
            return

        for r in range(5, -1, -1):
            if self.agent.env.board[r][col] == 0:
                self.agent.env.board[r][col] = self.current_player
                break

        self.update_ui(r, col)

        if self.agent.env.check_win(r, col):
            self.status_label.setText(f"Player {self.current_player} wins!")
            self.game_over = True
        elif self.agent.env.is_terminal():
            self.status_label.setText("It's a draw!")
            self.game_over = True
        else:
            self.current_player = 3 - self.current_player
            self.agent.env.current_player = 3 - self.agent.env.current_player
            self.play_ai_turn()

    def play_ai_turn(self):
        if self.game_over:
            return

        action = self.ai_select_move()
        if action is not None:
            row = self.update_game_state(action)
            self.update_ui(row, action)
            self.check_game_over(row, action)
            self.current_player = 3 - self.current_player
            self.agent.env.current_player = 3 - self.agent.env.current_player

    def ai_select_move(self):
        if isinstance(self.agent, HybridAgent):
            return self.agent.select_action(self.agent.env.board, epsilon=0)
        elif isinstance(self.agent, DQNAgent):
            return self.agent.select_action(self.agent.env.board, epsilon=0)
        elif isinstance(self.agent, DDQNAgent):
            return self.agent.select_action(self.agent.env.board, epsilon=0)
        elif isinstance(self.agent, C51Agent):
            return self.agent.select_action(self.agent.env.board, epsilon=0)
        elif isinstance(self.agent, PPOAgent):
            return self.agent.select_action(self.agent.env.board)
        elif isinstance(self.agent, A3CAgent):
            return self.agent.select_action(self.agent.env.board)
        elif isinstance(self.agent, SACAgent): # Add SACAgent to ai_select_move
            return self.agent.select_action(self.agent.env.board)
        else:
            self.status_label.setText("No agent is loaded.")
            return None

    def update_game_state(self, action):
        for r in range(5, -1, -1):
            if self.agent.env.board[r][action] == 0:
                self.agent.env.board[r][action] = self.current_player
                return r

    def update_ui(self, row, action):
        color = 'yellow' if self.current_player == 2 else 'red'
        self.board[row][action].setStyleSheet(f'background-color: {color};')

    def check_game_over(self, row, action):
        if self.agent.env.check_win(row, action):
            self.status_label.setText("AI wins!" if self.current_player == 2 else "Player 1 wins!")
            self.game_over = True
        elif self.agent.env.is_terminal():
            self.status_label.setText("It's a draw!")
            self.game_over = True

    def column_not_full(self, column):
        return self.board[0][column].styleSheet() == ""

    def select_agent(self):
        agent_type, ok = QInputDialog.getItem(self, "Select Agent Type",
                                            "Choose an agent:", ["DQN","DDQND", "Hybrid", "PPO", "A3C", "C51", "SAC"], 0, False) # Add "SAC" to agent list
        if ok and agent_type:
            if agent_type == "DQN":
                if self.current_player == 1:
                    self.agent = DQNAgent(ConnectFourEnv())
                    self.load_agent('saved_agents/dqn_agents_after_train.pth', player=2)
                else:
                    self.agent = DQNAgent(ConnectFourEnv())
                    self.load_agent('saved_agents/dqn_agents_after_train.pth', player=1)
            elif agent_type == "Hybrid":
                if self.current_player == 1:
                    self.agent = HybridAgent(ConnectFourEnv())
                    self.load_agent('saved_agents/hybrid_agents_after_train.pth', player=2)
                else:
                    self.agent = HybridAgent(ConnectFourEnv())
                    self.load_agent('saved_agents/hybrid_agents_after_train.pth', player=1)
            elif agent_type == "DDQND":
                if self.current_player == 1:
                    self.agent = DDQNAgent(ConnectFourEnv())
                    self.load_agent('saved_agents/ddqnd_agents_after_train.pth', player=2)
                else:
                    self.agent = DDQNAgent(ConnectFourEnv())
                    self.load_agent('saved_agents/ddqnd_agents_after_train.pth', player=1)
            elif agent_type == "PPO":
                if self.current_player == 1:
                    self.agent = PPOAgent(ConnectFourEnv())
                    self.load_agent('saved_agents/ppo_agents_after_train.pth', player=2)
                else:
                    self.agent = PPOAgent(ConnectFourEnv())
                    self.load_agent('saved_agents/ppo_agents_after_train.pth', player=1)
            elif agent_type == "A3C":
                if self.current_player == 1:
                    self.agent = A3CAgent(ConnectFourEnv())
                    self.load_agent('saved_agents/a3c_agents_after_train.pth', player=2)
                else:
                    self.agent = A3CAgent(ConnectFourEnv())
                    self.load_agent('saved_agents/a3c_agents_after_train.pth', player=1)
            elif agent_type == "C51":
                if self.current_player == 1:
                    self.agent = C51Agent(ConnectFourEnv())
                    self.load_agent('saved_agents/c51_agents_after_train.pth', player=2)
                else:
                    self.agent = C51Agent(ConnectFourEnv())
                    self.load_agent('saved_agents/c51_agents_after_train.pth', player=1)
            elif agent_type == "SAC": # Add SACAgent to select_agent
                if self.current_player == 1:
                    self.agent = SACAgent(ConnectFourEnv())
                    self.load_agent('saved_agents/sac_agents_after_train.pth', player=2)
                else:
                    self.agent = SACAgent(ConnectFourEnv())
                    self.load_agent('saved_agents/sac_agents_after_train.pth', player=1)

    def load_agent(self, filepath, player):
        try:
            # Load the agent based on its type and player
            if isinstance(self.agent, DQNAgent):
                # Load DQN agent
                checkpoint = torch.load(filepath)
                if player == 1:
                    self.agent.model.load_state_dict(checkpoint['model_state_dict_player1'])
                    self.agent.target_model.load_state_dict(checkpoint['target_model_state_dict_player1'])
                elif player == 2:
                    self.agent.model.load_state_dict(checkpoint['model_state_dict_player2'])
                    self.agent.target_model.load_state_dict(checkpoint['target_model_state_dict_player2'])
            elif isinstance(self.agent, DDQNAgent):
                # Load Hybrid agent
                checkpoint = torch.load(filepath)
                if player == 1:
                    self.agent.model.load_state_dict(checkpoint['model_state_dict_player1'])
                    self.agent.target_model.load_state_dict(checkpoint['target_model_state_dict_player1'])
                elif player == 2:
                    self.agent.model.load_state_dict(checkpoint['model_state_dict_player2'])
                    self.agent.target_model.load_state_dict(checkpoint['target_model_state_dict_player2'])
            elif isinstance(self.agent, HybridAgent):
                # Load Hybrid agent
                checkpoint = torch.load(filepath)
                if player == 1:
                    self.agent.model.load_state_dict(checkpoint['model_state_dict_player1'])
                    self.agent.target_model.load_state_dict(checkpoint['target_model_state_dict_player1'])
                elif player == 2:
                    self.agent.model.load_state_dict(checkpoint['model_state_dict_player2'])
                    self.agent.target_model.load_state_dict(checkpoint['target_model_state_dict_player2'])
            elif isinstance(self.agent, A3CAgent):
                # Load Hybrid agent
                checkpoint = torch.load(filepath)
                if player == 1:
                    self.agent.model.load_state_dict(checkpoint['model_state_dict_player1'])
                    self.agent.model.load_state_dict(checkpoint['model_state_dict_player1']) # Corrected duplicated line
                elif player == 2:
                    self.agent.model.load_state_dict(checkpoint['model_state_dict_player2'])
                    self.agent.model.load_state_dict(checkpoint['model_state_dict_player2']) # Corrected duplicated line
            elif isinstance(self.agent, PPOAgent):
                # Load Hybrid agent
                checkpoint = torch.load(filepath)
                if player == 1:
                    self.agent.model.load_state_dict(checkpoint['model_state_dict_player1'])
                    self.agent.model.load_state_dict(checkpoint['model_state_dict_player1']) # Corrected duplicated line
                elif player == 2:
                    self.agent.model.load_state_dict(checkpoint['model_state_dict_player2'])
                    self.agent.model.load_state_dict(checkpoint['model_state_dict_player2']) # Corrected duplicated line
            elif isinstance(self.agent, C51Agent):
                # Load Hybrid agent
                checkpoint = torch.load(filepath)
                if player == 1:
                    self.agent.model.load_state_dict(checkpoint['model_state_dict_player1'])
                    self.agent.model.load_state_dict(checkpoint['model_state_dict_player1']) # Corrected duplicated line
                elif player == 2:
                    self.agent.model.load_state_dict(checkpoint['model_state_dict_player2'])
                    self.agent.model.load_state_dict(checkpoint['model_state_dict_player2']) # Corrected duplicated line
            elif isinstance(self.agent, SACAgent): # Add SACAgent to load_agent
                # Load SAC agent
                checkpoint = torch.load(filepath)
                if player == 1:
                    self.agent.actor.load_state_dict(checkpoint['actor_state_dict_player1']) # Assuming actor and critic are in SACAgent
                    self.agent.critic1.load_state_dict(checkpoint['critic1_state_dict_player1'])
                    self.agent.critic2.load_state_dict(checkpoint['critic2_state_dict_player1'])
                elif player == 2:
                    self.agent.actor.load_state_dict(checkpoint['actor_state_dict_player2']) # Assuming actor and critic are in SACAgent
                    self.agent.critic1.load_state_dict(checkpoint['critic1_state_dict_player2'])
                    self.agent.critic2.load_state_dict(checkpoint['critic2_state_dict_player2'])

            # Display a success message
            self.status_label.setText(f"{type(self.agent).__name__} loaded successfully for Player {player}.")
            self.play_button.setDisabled(False)
            self.play_again_button.setDisabled(False)
        except FileNotFoundError:
            # Display an error message if the file is not found
            self.status_label.setText("Agent file not found.")
        except Exception as e:
            # Display an error message if loading fails
            self.status_label.setText(f"Failed to load agent: {str(e)}")

# Entry point of the application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ConnectFour()
    window.show()
    sys.exit(app.exec())