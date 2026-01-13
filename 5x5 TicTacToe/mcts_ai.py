import random
import math
import time
from copy import deepcopy

class Node:
    def __init__(self, board, game_logic, parent=None, move=None):
        self.board = deepcopy(board)
        self.game_logic = deepcopy(game_logic)
        self.parent = parent
        self.move = move  # The move that led to this state
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_moves = self.get_untried_moves()
    
    def get_untried_moves(self):
        """Get all valid moves that haven't been tried yet"""
        return self.board.get_empty_cells()
    
    def select_child(self):
        """Select the child with the highest UCB score"""
        # UCB1 formula: wins/visits + C * sqrt(ln(parent visits) / visits)
        C = 1.41  # Exploration parameter
        
        # Select child with highest UCB score
        return max(self.children, key=lambda child: 
                  (child.wins / child.visits if child.visits > 0 else float('inf')) + 
                  C * math.sqrt(math.log(self.visits) / child.visits if child.visits > 0 else float('inf')))
    
    def expand(self):
        """Expand the tree by adding a new child node"""
        if not self.untried_moves or self.game_logic.is_game_over():
            return None
        
        try:
            # Choose a random untried move
            move = random.choice(self.untried_moves)
            self.untried_moves.remove(move)
            
            # Create a new child node
            row, col = move
            new_board = deepcopy(self.board)
            new_game_logic = deepcopy(self.game_logic)
            
            # Make the move - ensure the cell is actually empty in the board state
            if new_board.is_cell_empty(row, col):
                new_board.make_move(row, col, new_game_logic.get_current_player())
                new_game_logic.make_move(row, col)
                
                # Create and add the new child node
                child = Node(new_board, new_game_logic, parent=self, move=move)
                self.children.append(child)
                return child
            else:
                # If the cell is not empty, remove it from untried_moves and try again
                return self.expand()
        except ValueError as e:
            # If there's an error (like cell not in list), try to recover
            print(f"Error in MCTS expand: {e}")
            # Refresh the untried moves list to match the actual board state
            self.untried_moves = []
            for r in range(self.board.size):
                for c in range(self.board.size):
                    if self.board.is_cell_empty(r, c):
                        self.untried_moves.append((r, c))
            
            # Try expanding again if we have moves
            if self.untried_moves:
                return self.expand()
            return None
    
    def simulate(self):
        """Simulate a random game from the current state until the end"""
        try:
            # Create copies to avoid modifying the original state
            sim_board = deepcopy(self.board)
            sim_game_logic = deepcopy(self.game_logic)
            
            # Maximum number of moves to prevent infinite loops
            max_moves = sim_board.size * sim_board.size
            move_count = 0
            
            # Simulate random moves until the game is over
            while not sim_game_logic.is_game_over() and move_count < max_moves:
                # Get all valid moves by checking the actual board state
                valid_moves = []
                for r in range(sim_board.size):
                    for c in range(sim_board.size):
                        if sim_board.is_cell_empty(r, c):
                            valid_moves.append((r, c))
                
                if not valid_moves:
                    break
                
                # Choose a random move
                row, col = random.choice(valid_moves)
                
                # Make the move with error handling
                if sim_board.is_cell_empty(row, col):
                    sim_board.make_move(row, col, sim_game_logic.get_current_player())
                    sim_game_logic.make_move(row, col)
                    move_count += 1
                else:
                    # Skip this move if the cell is already taken
                    continue
            
            # Return the result of the simulation
            winner = sim_game_logic.get_winner()
            return winner
            
        except Exception as e:
            print(f"Error in MCTS simulation: {e}")
            # Return a default result in case of error
            return None
    
    def backpropagate(self, result):
        """Backpropagate the result up the tree"""
        self.visits += 1
        
        # Update wins based on the result
        if result == 2:  # AI wins
            self.wins += 1
        elif result is None:  # Draw
            self.wins += 0.5
        
        # Backpropagate to parent
        if self.parent:
            self.parent.backpropagate(result)

class MCTSAI:
    def __init__(self):
        self.max_iterations = 500  # Reduced from 1000 to prevent excessive resource usage
        self.max_time = 0.8  # Reduced maximum thinking time to prevent freezing
        self.difficulty = "medium"  # Default difficulty
        self.timeout_occurred = False  # Flag to track timeout events
    
    def make_move(self, board, game_logic):
        """Make a move using Monte Carlo Tree Search"""
        # Adjust iterations based on difficulty
        if self.difficulty == "easy":
            self.max_iterations = 200
            self.max_time = 0.5
        elif self.difficulty == "medium":
            self.max_iterations = 500
            self.max_time = 1.0
        else:  # hard
            self.max_iterations = 1000
            self.max_time = 2.0
        
        try:
            # Create the root node
            root = Node(board, game_logic)
            
            # Start the timer
            start_time = time.time()
            iterations = 0
            self.timeout_occurred = False
            
            # Run MCTS for a fixed number of iterations or until time limit
            while iterations < self.max_iterations and (time.time() - start_time) < self.max_time:
                # Check if we're approaching the time limit
                if (time.time() - start_time) > (self.max_time * 0.9):
                    self.timeout_occurred = True
                    break
                    
                # Selection: Select a promising node
                node = root
                try:
                    while node.untried_moves == [] and node.children != [] and not node.game_logic.is_game_over():
                        node = node.select_child()
                except Exception as selection_error:
                    print(f"Selection error: {selection_error}")
                    break
                
                # Expansion: Expand the selected node
                if not node.game_logic.is_game_over() and node.untried_moves:
                    try:
                        node = node.expand()
                        if node is None:  # No expansion possible
                            continue
                    except Exception as expansion_error:
                        print(f"Expansion error: {expansion_error}")
                        continue
                
                # Simulation: Simulate a random game from the expanded node
                try:
                    result = node.simulate()
                    
                    # Backpropagation: Update statistics up the tree
                    node.backpropagate(result)
                except Exception as simulation_error:
                    print(f"Simulation error: {simulation_error}")
                    continue
                
                iterations += 1
            
            # If we have children, choose the best move
            if root.children:
                try:
                    # Select the child with the most visits
                    best_child = max(root.children, key=lambda child: child.visits)
                    return best_child.move
                except Exception as selection_error:
                    print(f"Best move selection error: {selection_error}")
                    # Fall through to fallback logic
            
            # Fallback: If no children or error occurred, find any valid move
            valid_moves = []
            for r in range(board.size):
                for c in range(board.size):
                    if board.is_cell_empty(r, c):
                        valid_moves.append((r, c))
            
            if valid_moves:
                # If timeout occurred, log it
                if self.timeout_occurred:
                    print("MCTS timeout occurred, using random valid move")
                return random.choice(valid_moves)
            else:
                # Board is full or in an invalid state
                raise ValueError("No valid moves available")
            
        except Exception as e:
            print(f"Error in MCTS make_move: {e}")
            # Fallback: Find any valid move on the board
            valid_moves = []
            for r in range(board.size):
                for c in range(board.size):
                    if board.is_cell_empty(r, c):
                        valid_moves.append((r, c))
            
            if valid_moves:
                return random.choice(valid_moves)
            else:
                # If no valid moves, return a default move that will be checked by the game logic
                return (0, 0)