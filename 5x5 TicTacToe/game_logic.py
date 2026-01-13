class GameLogic:
    def __init__(self, board):
        self.board = board
        self.current_player = 1  
        self.game_over = False
        self.winner = None
    
    def check_win(self, player):
        """
        Check if the specified player has won
        Win condition depends on board size:
        - 3×3 board: 3 in a row
        - 5×5 board: 4 in a row
        - 9×9 or larger: 5 in a row
        """
        board_size = self.board.size
        board_state = self.board.board
        
        # Determine win length based on board size
        if board_size == 3:
            win_length = 3
        elif board_size == 5:
            win_length = 5  # Changed from 4 to 5 as per requirement
        else:  # 9×9 or 11×11
            win_length = 5
        
        # Check horizontal wins
        for row in range(board_size):
            for col in range(board_size - win_length + 1):
                if all(board_state[row][col+i] == player for i in range(win_length)):
                    return True
        
        # Check vertical wins
        for row in range(board_size - win_length + 1):
            for col in range(board_size):
                if all(board_state[row+i][col] == player for i in range(win_length)):
                    return True
        
        # Check diagonal wins (top-left to bottom-right)
        for row in range(board_size - win_length + 1):
            for col in range(board_size - win_length + 1):
                if all(board_state[row+i][col+i] == player for i in range(win_length)):
                    return True
        
        # Check diagonal wins (bottom-left to top-right)
        for row in range(win_length - 1, board_size):
            for col in range(board_size - win_length + 1):
                if all(board_state[row-i][col+i] == player for i in range(win_length)):
                    return True
        
        return False
    
    def check_draw(self):
        """
        Check if the game is a draw
        """
        return self.board.is_board_full() and not self.check_win(1) and not self.check_win(2)
    
    def make_move(self, row, col):
        """
        Make a move for the current player
        Returns True if move is valid, False otherwise
        """
        if self.game_over:
            return False
        
        if self.board.make_move(row, col, self.current_player):
            # Check if the current player has won
            if self.check_win(self.current_player):
                self.game_over = True
                self.winner = self.current_player
            # Check if the game is a draw
            elif self.check_draw():
                self.game_over = True
                self.winner = None
            else:
                # Switch player
                self.current_player = 3 - self.current_player  
            
            return True
        
        return False
    
    def get_current_player(self):
        """
        Get the current player
        Returns 1 for human (X), 2 for AI (O)
        """
        return self.current_player
    
    def is_game_over(self):
        """
        Check if the game is over
        """
        return self.game_over
    
    def get_winner(self):
        """
        Get the winner of the game
        Returns 1 for human (X), 2 for AI (O), None for draw
        """
        return self.winner
    
    def reset(self):
        """
        Reset the game to initial state
        """
        self.board.reset()
        self.current_player = 1
        self.game_over = False
        self.winner = None