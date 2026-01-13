import tkinter as tk
from board import Board
from game_logic import GameLogic
from ai import AI
from mcts_ai import MCTSAI
from gui import GUI
from loading_screen import LoadingScreen
from difficulty_screen import DifficultyScreen
from mode_selection_screen import ModeSelectionScreen
from board_size_screen import BoardSizeScreen

def main():
    # Initialize the main window
    root = tk.Tk()
    root.title("Modern Tic Tac Toe AI")
    
    # Game mode and board size variables
    game_mode = None
    board_size = None
    
    # Function to start the game after all selections
    def start_game(difficulty, depth):
        # Create game components with selected board size
        board = Board()
        board.size = board_size  # Set the selected board size
        board.reset()  # Reset with new size
        
        game_logic = GameLogic(board)
        
        if game_mode == "human_vs_ai":
            # Human vs AI mode
            ai = AI()
            ai.difficulty = difficulty
            ai.max_depth = depth
            
            # Start the game with the selected difficulty
            gui = GUI(root, board, game_logic, ai)
        else:
            # AI vs AI mode - use medium difficulty to prevent excessive resource usage
            minimax_ai = AI()
            minimax_ai.difficulty = "medium"  # Use medium difficulty to reduce resource usage
            minimax_ai.max_depth = 3        # Reduced depth to prevent crashes
            
            mcts_ai = MCTSAI()
            mcts_ai.difficulty = "medium"    # Use medium difficulty to reduce resource usage
            
            # Start the game with two AIs at their full potential
            gui = GUI(root, board, game_logic, minimax_ai, mcts_ai)
    
    # Function to show difficulty screen
    def show_difficulty_screen():
        difficulty_screen = DifficultyScreen(root, start_game)
    
    # Function to handle board size selection
    def handle_board_size_selection(size):
        nonlocal board_size
        board_size = size
        if game_mode == "human_vs_ai":
            # Only show difficulty screen for human vs AI mode
            show_difficulty_screen()
        else:
            # For AI vs AI mode, start the game directly with medium difficulty
            start_game("medium", 3)
    
    # Function to handle game mode selection
    def handle_mode_selection(mode):
        nonlocal game_mode
        game_mode = mode
        board_size_screen = BoardSizeScreen(root, handle_board_size_selection)
    
    # Function to show mode selection screen after loading
    def show_mode_selection_screen():
        mode_selection_screen = ModeSelectionScreen(root, handle_mode_selection)
    
    # Start with loading screen
    loading_screen = LoadingScreen(root, show_mode_selection_screen)
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    main()