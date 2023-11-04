import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QGridLayout, QWidget, QLineEdit, QHBoxLayout
from PyQt6.QtCore import Qt
from dqn import DQNAgent, ConnectFourEnv  # Import DQNAgent and ConnectFourEnv from your original code
import random
import torch

class ConnectFour(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Connect Four")
        self.setGeometry(100, 100, 600, 600)
        self.initUI()
        self.dqn_agent = DQNAgent(ConnectFourEnv())
        self.current_player = 1
        self.game_over = False
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
                # Connect the button click event to the on_click method with row and col arguments
                button.clicked.connect(lambda _, row=row, col=col: self.on_click(row, col))
                self.grid.addWidget(button, row, col)
                self.board[row][col] = button

        self.status_label = QLabel()
        self.grid.addWidget(self.status_label, 6, 0, 1, 7, alignment=Qt.AlignmentFlag.AlignCenter)

        button_row_layout = QHBoxLayout()  # Create a horizontal layout for buttons and input fields

        self.load_button = QPushButton("Load Agent")
        self.load_button.clicked.connect(self.load_agent)
        button_row_layout.addWidget(self.load_button)

        self.play_button = QPushButton("Play Game")
        self.play_button.clicked.connect(self.start_game)
        button_row_layout.addWidget(self.play_button)
        self.play_button.setDisabled(True)

        self.grid.addLayout(button_row_layout, 7, 0, 1, 7)  # Add the button_row_layout in row 7

    def start_game(self):
        self.play_button.setDisabled(True)
        self.current_player = 1
        self.game_over = False
        self.status_label.setText("")
        self.game_state_history = []  # Clear the game state history
        for row in range(6):
            for col in range(7):
                self.board[row][col].setStyleSheet("")

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
            for r in range(5, -1, -1):
                if self.board[r][col].styleSheet() == "":
                    self.board[r][col].setStyleSheet('background-color: red;')
                    if self.check_win(r, col):
                        self.status_label.setText(f"Player 1 wins!")
                        self.game_over = True
                    elif self.check_draw():
                        self.status_label.setText("It's a draw!")
                        self.game_over = True
                    else:
                        self.current_player = 3 - self.current_player
                        if not self.game_over:
                            self.play_ai_turn()
                    break


    def play_ai_turn(self):
        if not self.game_over:
            action = self.dqn_agent.select_action(self.get_game_state(), epsilon=0.0)
            available_columns = [col for col in range(7) if self.column_not_full(col)]
            
            if (action is not None) and (action in available_columns):  # Make sure the selected action is valid.
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
    
    def check_draw(self):
        for col in range(7):
            if self.column_not_full(col):
                return False
        return True

    def load_agent(self):
        try:
            checkpoint = torch.load('saved_agents/dqn_agent_after_training.pth')
            self.dqn_agent.model.load_state_dict(checkpoint['model_state_dict'])
            self.dqn_agent.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.dqn_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.status_label.setText("Agent loaded successfully.")

            # Enable the "Play Game" button
            self.play_button.setDisabled(False)

        except FileNotFoundError:
            self.status_label.setText("Agent file not found.")
        except Exception as e:
            self.status_label.setText("Failed to load agent: " + str(e))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ConnectFour()
    window.show()
    sys.exit(app.exec())
