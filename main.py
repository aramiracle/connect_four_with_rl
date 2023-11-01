import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QGridLayout, QWidget, QLineEdit, QHBoxLayout
from PyQt5.QtCore import Qt
from dqn import DQNAgent, ConnectFourEnv  # Import DQNAgent and ConnectFourEnv from your original code
import random
import time

class ConnectFour(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Connect Four")
        self.setGeometry(100, 100, 600, 600)
        self.initUI()
        self.dqn_agent = DQNAgent(ConnectFourEnv())
        self.current_player = 1
        self.game_over = False
        self.is_training = False
        self.game_state_history = []

    def initUI(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.grid = QGridLayout()
        self.central_widget.setLayout(self.grid)
        self.board = [[0] * 7 for _ in range(6)]

        for row in range(6):
            for col in range(7):
                button = QPushButton()
                button.setFixedSize(80, 80)
                button.clicked.connect(lambda _, row=row, col=col: self.on_click(row, col))
                self.grid.addWidget(button, row, col)
                self.board[row][col] = button

        self.status_label = QLabel()
        self.grid.addWidget(self.status_label, 6, 0, 1, 7, alignment=Qt.AlignCenter)

        button_row_layout = QHBoxLayout()  # Create a horizontal layout for buttons and input fields

        self.episodes_label = QLabel("Num Episodes:")
        button_row_layout.addWidget(self.episodes_label)
        self.episodes_input = QLineEdit("100")
        button_row_layout.addWidget(self.episodes_input)

        self.train_button = QPushButton("Train AI")
        self.train_button.clicked.connect(self.start_training)
        button_row_layout.addWidget(self.train_button)

        self.play_button = QPushButton("Play Game")
        self.play_button.clicked.connect(self.start_game)
        button_row_layout.addWidget(self.play_button)
        self.play_button.setDisabled(True)

        self.grid.addLayout(button_row_layout, 7, 0, 1, 7)  # Add the button_row_layout in row 7

    def start_training(self):
        if not self.is_training:
            num_episodes = int(self.episodes_input.text())
            self.is_training = True
            self.train_button.setDisabled(True)
            self.dqn_agent.train(num_episodes=num_episodes)
            self.is_training = False
            self.play_button.setDisabled(False)

    def start_game(self):
        if not self.is_training:
            self.play_button.setDisabled(True)
            self.current_player = 1
            self.game_over = False
            self.status_label.setText("")
            self.game_state_history = []  # Clear the game state history
            for row in range(6):
                for col in range(7):
                    self.board[row][col].setStyleSheet("")

    def replay_game(self):
        if self.game_state_history:
            self.play_button.setDisabled(True)
            self.replay_button.setDisabled(True)
            for game_state in self.game_state_history:
                self.update_board_from_game_state(game_state)
                time.sleep(1)  # Optional delay to make the replay more visible
            self.play_button.setDisabled(False)
            self.replay_button.setDisabled(False)

    def update_board_from_game_state(self, game_state):
        for row in range(6):
            for col in range(7):
                if game_state[row][col] == 1:
                    self.board[row][col].setStyleSheet('background-color: red;')
                elif game_state[row][col] == 2:
                    self.board[row][col].setStyleSheet('background-color: yellow;')
                else:
                    self.board[row][col].setStyleSheet('')

    def on_click(self, row, col):
        if not self.game_over:
            # Find the lowest available row in the selected column
            for r in range(5, -1, -1):
                if self.board[r][col].styleSheet() == "":
                    self.board[r][col].setStyleSheet('background-color: red;')
                    if self.check_win(r, col):
                        self.status_label.setText(f"Player 1 wins!")
                        self.game_over = True
                    else:
                        self.current_player = 3 - self.current_player
                        if not self.game_over:
                            self.play_ai_turn()
                    break

    def play_ai_turn(self):
        if not self.game_over:
            action = self.dqn_agent.select_action(self.get_game_state(), epsilon=0.0)
            
            if action is not None:  # Make sure the selected action is valid.
                if self.column_not_full(action):
                    for row in range(5, -1, -1):
                        if self.board[row][action].styleSheet() == "":
                            self.board[row][action].setStyleSheet('background-color: yellow;')
                            if self.check_win(row, action):
                                self.status_label.setText(f"AI wins!")
                                self.game_over = True
                            else:
                                self.current_player = 3 - self.current_player
                            break
            else:
                # If action is None or the column is full, choose a random legal move.
                available_columns = [col for col in range(7) if self.column_not_full(col)]
                if available_columns:
                    action = random.choice(available_columns)
                    for row in range(5, -1, -1):
                        if self.board[row][action].styleSheet() == "":
                            self.board[row][action].setStyleSheet('background-color: yellow;')
                            if self.check_win(row, action):
                                self.status_label.setText(f"AI wins!")
                                self.game_over = True
                            else:
                                self.current_player = 3 - self.current_player
                            break

    def column_not_full(self, column):
        return self.board[0][column].styleSheet() == ""
            

    def get_game_state(self):
        game_state = [[0] * 7 for _ in range(6)]
        for row in range(6):
            for col in range(7):
                if self.board[row][col].styleSheet() == 'background-color: red;':
                    game_state[row][col] = 1
                elif self.board[row][col].styleSheet() == 'background-color: yellow;':
                    game_state[row][col] = 2
        return game_state

    def check_win(self, row, col):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, 4):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < 6 and 0 <= c < 7 and self.board[r][c].styleSheet() == self.board[row][col].styleSheet():
                    count += 1
                else:
                    break
            for i in range(1, 4):
                r, c = row - dr * i, col - dc * i
                if 0 <= r < 6 and 0 <= c < 7 and self.board[r][c].styleSheet() == self.board[row][col].styleSheet():
                    count += 1
                else:
                    break
            if count >= 4:
                return True
        return False

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ConnectFour()
    window.show()
    sys.exit(app.exec_())