class AI:
    def __init__(self):
        self.max_depth = 3 
        self.difficulty = "medium" 
    
    def make_move(self, board, game_logic):
        """
        Make a move for AI based on difficulty level
        - Hard: Always makes the best move
        - Medium: Occasionally makes a suboptimal move
        - Easy: Frequently makes suboptimal moves
        """
        # Get all possible moves with their scores
        moves_with_scores = []
        alpha = float('-inf')
        beta = float('inf')
        
        # Try each empty cell and calculate its score
        for row, col in board.get_empty_cells():
            board.make_move(row, col, 2) 
            
            # Calculate score using minimax
            score = self.minimax(board, game_logic, 0, False, alpha, beta)
            
            # Undo the move
            board.undo_move(row, col)
            
            # Store the move and its score
            moves_with_scores.append(((row, col), score))
            
            # Update alpha for future pruning
            alpha = max(alpha, score)
        
        # Sort moves by score (best moves first)
        moves_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Choose move based on difficulty
        import random
        
        if self.difficulty == "hard" or len(moves_with_scores) <= 1:
            # Hard: Always choose the best move
            return moves_with_scores[0][0]
        
        elif self.difficulty == "medium":
            # Medium: 90% best move, 10% second best or worse
            if random.random() < 0.9 or len(moves_with_scores) <= 2:
                return moves_with_scores[0][0]  # Best move
            else:
                suboptimal_index = random.randint(1, min(3, len(moves_with_scores) - 1))
                return moves_with_scores[suboptimal_index][0]
        
        else:  # Easy mode
            # Easy: 70% best move, 30% suboptimal move
            if random.random() < 0.7 or len(moves_with_scores) <= 2:
                return moves_with_scores[0][0]  # Best move
            else:
            
                max_index = min(len(moves_with_scores) - 1, 4)  
                weights = [1] + [2] * max_index 
                suboptimal_index = random.choices(range(max_index + 1), weights=weights[:max_index+1])[0]
                return moves_with_scores[suboptimal_index][0]
    
    def minimax(self, board, game_logic, depth, is_maximizing, alpha, beta):
        """
        Minimax algorithm with Alpha-Beta pruning
        """
        # Check if AI wins
        if game_logic.check_win(2):
            return 100 - depth
        
        # Check if human wins
        if game_logic.check_win(1):
            return -100 + depth
        
        # Check if it's a draw
        if board.is_board_full():
            return 0
        
        # Check if maximum depth is reached
        if depth >= self.max_depth:
            return self.evaluate_board(board)
        
        if is_maximizing:
            # AI's turn (maximizing)
            best_score = float('-inf')
            
            for row, col in board.get_empty_cells():
                # Make the move
                board.make_move(row, col, 2)  # 2 represents AI (O)
                
                # Recursively calculate score
                score = self.minimax(board, game_logic, depth + 1, False, alpha, beta)
                
                # Undo the move
                board.undo_move(row, col)
                
                # Update best score
                best_score = max(best_score, score)
                
                # Alpha-Beta pruning
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            
            return best_score
        else:
            # Human's turn (minimizing)
            best_score = float('inf')
            
            for row, col in board.get_empty_cells():
                # Make the move
                board.make_move(row, col, 1)  # 1 represents human (X)
                
                # Recursively calculate score
                score = self.minimax(board, game_logic, depth + 1, True, alpha, beta)
                
                # Undo the move
                board.undo_move(row, col)
                
                # Update best score
                best_score = min(best_score, score)
                
                # Alpha-Beta pruning
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
            
            return best_score
    
    def evaluate_board(self, board):
        """
        Evaluate the current board state
        Positive score favors AI, negative score favors human
        """
        score = 0
        board_state = board.board
        board_size = board.size
        
        # Determine win length based on board size
        if board_size == 3:
            win_length = 3
        elif board_size == 5:
            win_length = 4
        else:  # 9×9 or 11×11
            win_length = 5
        
        # Check rows, columns, and diagonals for potential wins
        for row in range(board_size):
            for col in range(board_size - win_length + 1):
                window = [board_state[row][col+i] for i in range(win_length)]
                score += self.evaluate_window(window, win_length)

        for row in range(board_size - win_length + 1):
            for col in range(board_size):
                window = [board_state[row+i][col] for i in range(win_length)]
                score += self.evaluate_window(window, win_length)
        
        for row in range(board_size - win_length + 1):
            for col in range(board_size - win_length + 1):
                window = [board_state[row+i][col+i] for i in range(win_length)]
                score += self.evaluate_window(window, win_length)
        
        for row in range(win_length - 1, board_size):
            for col in range(board_size - win_length + 1):
                window = [board_state[row-i][col+i] for i in range(win_length)]
                score += self.evaluate_window(window, win_length)
        
        return score
    
    def evaluate_window(self, window, win_length):
        """
        Evaluate a window of cells based on the win length
        """
        score = 0
        ai_count = window.count(2)  # Count AI pieces
        human_count = window.count(1)  # Count human pieces
        empty_count = window.count(0)  # Count empty cells
        
        # Score based on piece counts in the window
        if win_length == 3:
            # 3×3 board scoring
            if ai_count == 2 and empty_count == 1:
                score += 50  # AI is one move away from winning
            elif ai_count == 1 and empty_count == 2:
                score += 10  # AI has a good opportunity
            
            if human_count == 2 and empty_count == 1:
                score -= 50  # Human is one move away from winning (block!)
            elif human_count == 1 and empty_count == 2:
                score -= 10  # Human has a good opportunity (block!)
        
        elif win_length == 4:
            # 5×5 board scoring
            if ai_count == 3 and empty_count == 1:
                score += 50  # AI is one move away from winning
            elif ai_count == 2 and empty_count == 2:
                score += 10  # AI has a good opportunity
            elif ai_count == 1 and empty_count == 3:
                score += 5  # AI has some potential
            
            if human_count == 3 and empty_count == 1:
                score -= 50  # Human is one move away from winning (block!)
            elif human_count == 2 and empty_count == 2:
                score -= 10  # Human has a good opportunity (block!)
            elif human_count == 1 and empty_count == 3:
                score -= 5  # Human has some potential (block!)
        
        else:  # win_length == 5 (9×9 or 11×11 board)
            if ai_count == 4 and empty_count == 1:
                score += 50  # AI is one move away from winning
            elif ai_count == 3 and empty_count == 2:
                score += 10  # AI has a good opportunity
            elif ai_count == 2 and empty_count == 3:
                score += 5  # AI has some potential
            
            if human_count == 4 and empty_count == 1:
                score -= 50  # Human is one move away from winning (block!)
            elif human_count == 3 and empty_count == 2:
                score -= 10  # Human has a good opportunity (block!)
            elif human_count == 2 and empty_count == 3:
                score -= 5  # Human has some potential (block!)
        
        return score