import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QGridLayout, QWidget
from PyQt5.QtCore import Qt

class ConnectFour(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Connect Four")
        self.setGeometry(100, 100, 600, 600)

        self.initUI()

    def initUI(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.grid = QGridLayout()
        self.central_widget.setLayout(self.grid)

        # Create a 2D array to represent the game board with buttons
        self.board = [[0] * 7 for _ in range(6)]

        # Initialize the current player (1 for Player 1, 2 for Player 2)
        self.current_player = 1

        for row in range(6):
            for col in range(7):
                # Create a button for each cell in the grid
                button = QPushButton()
                button.setFixedSize(80, 80)
                # Connect the button's click event to the on_click method with row and col arguments
                button.clicked.connect(lambda _, row=row, col=col: self.on_click(row, col))
                self.grid.addWidget(button, row, col)
                self.board[row][col] = button

        # Create a status label to display the game's status
        self.status_label = QLabel()
        self.grid.addWidget(self.status_label, 6, 0, 1, 7, alignment=Qt.AlignCenter)

    def on_click(self, row, col):
        for r in reversed(range(6)):
            # Check if the cell is empty (no background color)
            if not self.board[r][col].styleSheet():
                # Set the background color of the cell to red or yellow based on the current player
                self.board[r][col].setStyleSheet('background-color: red;' if self.current_player == 1 else 'background-color: yellow;')
                if self.check_win(r, col):
                    # If a player wins, display a message in the status label
                    self.status_label.setText(f"Player {self.current_player} wins!")
                else:
                    # Switch to the next player's turn
                    self.current_player = 3 - self.current_player  # Switch players (1 to 2, 2 to 1)

                    # After switching players, check for a draw (board is full)
                    if all(self.board[0][i].styleSheet() for i in range(7)):
                        self.status_label.setText("It's a draw!")

                break


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
