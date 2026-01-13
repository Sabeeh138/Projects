import tkinter as tk
from tkinter import messagebox
import time
import random
from tkinter import font as tkfont
import math

class GUI:
    def __init__(self, root, board, game_logic, ai, second_ai=None):
        self.root = root
        self.board = board
        self.game_logic = game_logic
        self.ai = ai
        
     
        # Modern dark theme color palette
        self.bg_color = '#0f0f1a'  # Deep blue-black background
        self.header_color = '#1a1a2e'  # Slightly lighter than background
        self.btn_color = '#16213e'  # Dark blue for buttons
        self.btn_hover_color = '#1e2a4a'  # Lighter blue for hover
        self.human_color = '#00cec9'  # Teal for human player
        self.ai_color = '#fd79a8'  # Pink for AI player
        self.win_line_color = '#6c5ce7'  # Purple for win highlight
        self.text_color = '#ffffff'  # White text
        self.accent_color = '#6c5ce7'  # Vibrant purple accent
        
       
        self.player_score = 0
        self.ai_score = 0
        self.winning_cells = []
        
        # Performance metrics for AI vs AI mode
        self.minimax_metrics = {
            'total_moves': 0,
            'total_time': 0,
            'avg_move_time': 0,
            'max_move_time': 0,
            'min_move_time': float('inf'),
            'current_game_moves': 0,
            'current_game_time': 0
        }
        
        self.mcts_metrics = {
            'total_moves': 0,
            'total_time': 0,
            'avg_move_time': 0,
            'max_move_time': 0,
            'min_move_time': float('inf'),
            'current_game_moves': 0,
            'current_game_time': 0
        }
        
        self.current_move_start_time = 0
        
        self.root.configure(bg=self.bg_color)
        self.root.resizable(True, True)
        self.root.title("Modern Tic Tac Toe")
        
        
        self.title_font = tkfont.Font(family="Helvetica", size=24, weight="bold")
        self.status_font = tkfont.Font(family="Helvetica", size=14)
        self.button_font = tkfont.Font(family="Helvetica", size=16, weight="bold")
        
        # Create frames
        self.header_frame = tk.Frame(root, bg=self.header_color, padx=20, pady=15)
        self.header_frame.pack(fill=tk.X, pady=0)
        
        self.score_frame = tk.Frame(root, bg=self.bg_color, padx=20, pady=10)
        self.score_frame.pack(fill=tk.X)
        
        self.board_frame = tk.Frame(root, bg=self.bg_color, padx=20, pady=15)
        self.board_frame.pack(pady=10)
        
        self.footer_frame = tk.Frame(root, bg=self.bg_color, padx=20, pady=15)
        self.footer_frame.pack(pady=10, fill=tk.X)
        
        
        # Update title to reflect the selected board size
        board_size_text = f"{board.size}×{board.size}"
        self.title_label = tk.Label(self.header_frame, text=f"{board_size_text} Tic Tac Toe", 
                                  font=self.title_font, bg=self.header_color, fg=self.text_color)
        self.title_label.pack(pady=10)
        
        
        self.score_frame_left = tk.Frame(self.score_frame, bg=self.bg_color)
        self.score_frame_left.pack(side=tk.LEFT, expand=True)
        
        self.score_frame_right = tk.Frame(self.score_frame, bg=self.bg_color)
        self.score_frame_right.pack(side=tk.RIGHT, expand=True)
        
        # Check if we're in AI vs AI mode
        self.ai_vs_ai_mode = second_ai is not None
        self.second_ai = second_ai
        
        if self.ai_vs_ai_mode:
            self.player_score_label = tk.Label(self.score_frame_left, text="MINIMAX: 0", 
                                             font=self.status_font, bg=self.bg_color, fg=self.human_color)
            self.player_score_label.pack(side=tk.LEFT, padx=15)
            
            self.ai_score_label = tk.Label(self.score_frame_right, text="MCTS: 0", 
                                         font=self.status_font, bg=self.bg_color, fg=self.ai_color)
            self.ai_score_label.pack(side=tk.RIGHT, padx=15)
        else:
            self.player_score_label = tk.Label(self.score_frame_left, text="YOU: 0", 
                                             font=self.status_font, bg=self.bg_color, fg=self.human_color)
            self.player_score_label.pack(side=tk.LEFT, padx=15)
            
            self.ai_score_label = tk.Label(self.score_frame_right, text="AI: 0", 
                                         font=self.status_font, bg=self.bg_color, fg=self.ai_color)
            self.ai_score_label.pack(side=tk.RIGHT, padx=15)
        
        
        self.status_frame = tk.Frame(self.score_frame, bg=self.bg_color)
        self.status_frame.pack(pady=10)
        
        # Set initial status text based on mode
        initial_status = "YOUR TURN" if not self.ai_vs_ai_mode else "MINIMAX'S TURN"
        self.status_label = tk.Label(self.status_frame, text=initial_status, 
                                    font=self.status_font, bg=self.bg_color, fg=self.text_color,
                                    padx=15, pady=5)
        self.status_label.pack()
        
        
        self.pulse_direction = 1
        self.pulse_intensity = 0
        self.pulse_status_label()
        
        
        self.buttons = []
        for row in range(board.size):
            button_row = []
            for col in range(board.size):
                button = tk.Button(self.board_frame, text="", width=3, height=1, 
                                  font=self.button_font, bg=self.btn_color, fg=self.text_color,
                                  activebackground=self.btn_hover_color, activeforeground=self.text_color,
                                  bd=0, relief=tk.FLAT, cursor="hand2",
                                  command=lambda r=row, c=col: self.handle_button_click(r, c))
                button.grid(row=row, column=col, padx=4, pady=4, ipadx=8, ipady=8)
                
               
                button.bind("<Enter>", lambda event, btn=button: self.on_button_hover(btn))
                button.bind("<Leave>", lambda event, btn=button: self.on_button_leave(btn))
                
                button_row.append(button)
            self.buttons.append(button_row)
        
        # Create footer with buttons
        self.button_frame = tk.Frame(self.footer_frame, bg=self.bg_color)
        self.button_frame.pack()
        
        
        self.reset_button = tk.Button(self.button_frame, text="NEW GAME", font=self.status_font,
                                     command=self.reset_game, bg=self.human_color, fg=self.text_color,
                                     activebackground="#0097a7", activeforeground=self.text_color,
                                     relief=tk.FLAT, padx=20, pady=8, cursor="hand2")
        self.reset_button.pack(side=tk.LEFT, padx=15)
        
        
        self.difficulty_level = 3  
        self.difficulty_button = tk.Button(self.button_frame, text="DIFFICULTY: MEDIUM", 
                                         font=self.status_font, command=self.change_difficulty,
                                         bg=self.accent_color, fg=self.text_color, 
                                         activebackground="#5e35b1", activeforeground=self.text_color,
                                         relief=tk.FLAT, padx=20, pady=8, cursor="hand2")
        
        # Only show difficulty button in human vs AI mode
        if not self.ai_vs_ai_mode:
            self.difficulty_button.pack(side=tk.LEFT, padx=15)
        
        # If in AI vs AI mode, start the game automatically after a short delay
        if self.ai_vs_ai_mode:
            self.root.after(1500, self.make_ai_move)
    
    def pulse_status_label(self):
        """Create a pulsing effect for the status label"""
        
        self.pulse_intensity += 0.05 * self.pulse_direction
        if self.pulse_intensity >= 1.0:
            self.pulse_direction = -1
        elif self.pulse_intensity <= 0.0:
            self.pulse_direction = 1
            
        
        base_color = self.human_color if self.game_logic.get_current_player() == 1 else self.ai_color
        
    
        if not self.game_logic.is_game_over():
            self.status_label.config(fg=base_color)
        
        self.root.after(50, self.pulse_status_label)
    
    def on_button_hover(self, button):
        """Add hover effect to buttons"""
        if button["state"] != tk.DISABLED:
            button.config(bg=self.btn_hover_color)
           
            current_player = self.game_logic.get_current_player()
            glow_color = self.human_color if current_player == 1 else self.ai_color
            button.config(highlightbackground=glow_color, highlightthickness=2)
    
    def on_button_leave(self, button):
        """Remove hover effect from buttons"""
        if button["state"] != tk.DISABLED:
            button.config(bg=self.btn_color, highlightthickness=0)
    
    def change_difficulty(self):
        """Change AI difficulty level with visual feedback"""
        levels = [(1, "EASY", "#4caf50"), (3, "MEDIUM", "#ff9800"), (5, "HARD", "#f44336")]
        current_index = next((i for i, (level, _, _) in enumerate(levels) if level == self.difficulty_level), 0)
        next_index = (current_index + 1) % len(levels)
        self.difficulty_level, difficulty_name, color = levels[next_index]
        
        self.ai.max_depth = self.difficulty_level
        self.ai.difficulty = difficulty_name.lower()
        
        original_bg = self.difficulty_button["bg"]
        self.difficulty_button.config(bg=color)
        self.root.after(200, lambda: self.difficulty_button.config(bg=self.accent_color))
        
        self.difficulty_button.config(text=f"DIFFICULTY: {difficulty_name}")
    
    def highlight_winning_cells(self):
        """Highlight the winning cells with animation"""
        if not self.winning_cells:
            return
            
        def animate_cell(index):
            if index < len(self.winning_cells):
                row, col = self.winning_cells[index]
                button = self.buttons[row][col]
                
                # Flash effect
                for i in range(3):
                    button.config(bg=self.win_line_color)
                    self.root.update()
                    time.sleep(0.05)
                    button.config(bg=self.btn_color)
                    self.root.update()
                    time.sleep(0.05)
                
               
                button.config(bg=self.win_line_color)
                
                
                self.root.after(100, lambda: animate_cell(index + 1))
        
        
        animate_cell(0)
    
    def handle_button_click(self, row, col):
        """
        Handle button click event
        """
        # In AI vs AI mode, button clicks are disabled
        if self.ai_vs_ai_mode:
            return
            
        if self.game_logic.is_game_over() or self.game_logic.get_current_player() != 1:
            return
        
        if self.game_logic.make_move(row, col):
          
            self.animate_move(row, col, "X", self.human_color)
            
            if self.game_logic.is_game_over():
                self.find_winning_cells()
                self.highlight_winning_cells()
                self.show_game_result()
                return
            
        
            self.animate_thinking()
            self.root.update() 
            
            self.root.after(random.randint(300, 800))
            
            self.make_ai_move()
    
    def animate_thinking(self):
        """Animate the AI thinking status with dots"""
        dots = [".  ", ".. ", "..."]
        for i in range(3):
            self.status_label.config(text=f"AI IS THINKING{dots[i]}", fg=self.ai_color)
            self.root.update()
            time.sleep(0.2)
    
    def animate_move(self, row, col, symbol, color):
        """Animate a move with enhanced visual effects"""
        button = self.buttons[row][col]
        button.config(text="", state=tk.DISABLED)
        
       
        for i in range(5):
            size = i * 0.2
            alpha = 1.0 - (i * 0.2)
            button.config(bg=color)
            self.root.update()
            time.sleep(0.05)
            button.config(bg=self.btn_color)
            self.root.update()
            time.sleep(0.02)
        
        for opacity in range(0, 11, 2):
           
            opacity_factor = opacity / 10.0
            button.config(text=symbol, disabledforeground=color, bg=self.btn_color)
            self.root.update()
            time.sleep(0.03)
        
        
        button.config(text=symbol, disabledforeground=color, bg=self.btn_color)
    
    def find_winning_cells(self):
        """Find the cells that form the winning line"""
        self.winning_cells = []
        winner = self.game_logic.get_winner()
        if winner is None:  
            return
            
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
                if all(board_state[row][col+i] == winner for i in range(win_length)):
                    self.winning_cells = [(row, col+i) for i in range(win_length)]
                    return
        
        # Check vertical wins
        for row in range(board_size - win_length + 1):
            for col in range(board_size):
                if all(board_state[row+i][col] == winner for i in range(win_length)):
                    self.winning_cells = [(row+i, col) for i in range(win_length)]
                    return
        
        # Check diagonal wins (top-left to bottom-right)
        for row in range(board_size - win_length + 1):
            for col in range(board_size - win_length + 1):
                if all(board_state[row+i][col+i] == winner for i in range(win_length)):
                    self.winning_cells = [(row+i, col+i) for i in range(win_length)]
                    return
        
        # Check diagonal wins (bottom-left to top-right)
        for row in range(win_length - 1, board_size):
            for col in range(board_size - win_length + 1):
                if all(board_state[row-i][col+i] == winner for i in range(win_length)):
                    self.winning_cells = [(row-i, col+i) for i in range(win_length)]
                    return
    
    def make_ai_move(self):
        """
        Make a move for the AI with enhanced error handling and resource management
        """
        try:
            # Check if game is already over to prevent unnecessary processing
            if self.game_logic.is_game_over():
                return
                
            current_player = self.game_logic.get_current_player()
            
            # Determine which AI to use based on the current player and game mode
            if self.ai_vs_ai_mode:
                # In AI vs AI mode, player 1 uses minimax AI and player 2 uses MCTS AI
                current_ai = self.ai if current_player == 1 else self.second_ai
                ai_name = "MINIMAX" if current_player == 1 else "MCTS"
                symbol = "X" if current_player == 1 else "O"
                color = self.human_color if current_player == 1 else self.ai_color
                next_status = "MCTS'S TURN" if current_player == 1 else "MINIMAX'S TURN"
                
                # Start timing the AI move
                self.current_move_start_time = time.time()
            else:
                # In Human vs AI mode, only player 2 (AI) makes moves here
                current_ai = self.ai
                ai_name = "AI"
                symbol = "O"
                color = self.ai_color
                next_status = "YOUR TURN"
            
            # Update status to show which AI is thinking
            self.status_label.config(text=f"{ai_name} IS THINKING...", fg=color)
            self.root.update()
            
            # Get the AI's move with enhanced error handling
            try:
                # Add a small delay to prevent UI freezing and reduce CPU usage
                self.root.after(50)
                self.root.update()
                
                # Get AI move with a timeout mechanism
                ai_row, ai_col = current_ai.make_move(self.board, self.game_logic)
                
                # If in AI vs AI mode, record the move time
                if self.ai_vs_ai_mode:
                    move_time = time.time() - self.current_move_start_time
                    metrics = self.minimax_metrics if current_player == 1 else self.mcts_metrics
                    
                    # Update metrics
                    metrics['total_moves'] += 1
                    metrics['current_game_moves'] += 1
                    metrics['total_time'] += move_time
                    metrics['current_game_time'] += move_time
                    metrics['max_move_time'] = max(metrics['max_move_time'], move_time)
                    metrics['min_move_time'] = min(metrics['min_move_time'], move_time)
                    metrics['avg_move_time'] = metrics['total_time'] / metrics['total_moves']
                
                # Verify the move is valid
                if not (0 <= ai_row < self.board.size and 0 <= ai_col < self.board.size):
                    print(f"Invalid move coordinates: ({ai_row}, {ai_col})")
                    raise ValueError(f"Invalid move coordinates: ({ai_row}, {ai_col})")
                
                if not self.board.is_cell_empty(ai_row, ai_col):
                    print(f"Cell ({ai_row}, {ai_col}) is not empty, finding alternative move")
                    # Find a valid move instead
                    valid_moves = []
                    for r in range(self.board.size):
                        for c in range(self.board.size):
                            if self.board.is_cell_empty(r, c):
                                valid_moves.append((r, c))
                    
                    if valid_moves:
                        ai_row, ai_col = random.choice(valid_moves)
                        print(f"Selected alternative move: ({ai_row}, {ai_col})")
                    else:
                        # No valid moves left
                        print("No valid moves available")
                        return
            except Exception as e:
                print(f"Error getting {ai_name} move: {e}")
                # Find any valid move as a fallback
                valid_moves = []
                for r in range(self.board.size):
                    for c in range(self.board.size):
                        if self.board.is_cell_empty(r, c):
                            valid_moves.append((r, c))
                
                if valid_moves:
                    ai_row, ai_col = random.choice(valid_moves)
                    print(f"Using fallback move: ({ai_row}, {ai_col})")
                else:
                    # No valid moves left
                    print("No valid moves available for fallback")
                    return
            
            # Make the move on the board
            if self.game_logic.make_move(ai_row, ai_col):
                # Animate the move with error handling
                try:
                    self.animate_move(ai_row, ai_col, symbol, color)
                except Exception as anim_error:
                    print(f"Animation error: {anim_error}")
                    # Update the button directly if animation fails
                    self.buttons[ai_row][ai_col].config(text=symbol, disabledforeground=color, bg=self.btn_color, state=tk.DISABLED)
                
                # Check if the game is over
                if self.game_logic.is_game_over():
                    try:
                        self.find_winning_cells()
                        self.highlight_winning_cells()
                        self.show_game_result()
                    except Exception as end_error:
                        print(f"Game end handling error: {end_error}")
                        # Simple fallback for game end
                        winner = self.game_logic.get_winner()
                        winner_text = "MINIMAX WINS!" if winner == 1 else "MCTS WINS!" if winner == 2 else "DRAW!"
                        self.status_label.config(text=winner_text)
                else:
                    # Update status for next player
                    self.status_label.config(text=next_status, fg=self.human_color if current_player == 2 else self.ai_color)
                    
                    # If in AI vs AI mode, schedule the next AI move after a delay
                    if self.ai_vs_ai_mode:
                        try:
                            # Add a thinking animation for the next AI
                            next_ai_name = "MCTS" if current_player == 1 else "MINIMAX"
                            self.status_label.config(text=f"{next_ai_name} IS THINKING...", 
                                                   fg=self.ai_color if current_player == 1 else self.human_color)
                            self.root.update()
                            
                            # Use a shorter delay to reduce resource usage
                            # Schedule the next AI move with a delay for better visualization
                            self.root.after(800, self.make_ai_move)
                        except Exception as schedule_error:
                            print(f"Error scheduling next AI move: {schedule_error}")
                            # Try a simpler approach if the normal scheduling fails
                            try:
                                self.root.after(1000, self.make_ai_move)
                            except:
                                print("Critical error in AI scheduling")
        except Exception as e:
            print(f"Error in make_ai_move: {e}")
            # Try to recover and continue the game
            if self.ai_vs_ai_mode and not self.game_logic.is_game_over():
                try:
                    # Schedule another attempt after a delay with reduced frequency
                    self.root.after(1500, self.make_ai_move)
                except Exception as recovery_error:
                    print(f"Recovery attempt failed: {recovery_error}")
                    # Last resort - try to reset the game
                    try:
                        self.reset_game()
                    except:
                        pass
    
    def show_game_result(self):
        """
        Show the game result with enhanced visuals
        """
        winner = self.game_logic.get_winner()
        
        result_window = tk.Toplevel(self.root)
        result_window.configure(bg=self.bg_color)
        result_window.title("Game Result")
        # Increase height for AI vs AI mode to accommodate scrollable metrics
        result_window.geometry("450x400" if self.ai_vs_ai_mode else "300x200")
        result_window.resizable(True, True)  # Allow resizing
        result_window.transient(self.root) 
        result_window.grab_set()  
        
        result_window.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, 
                                         self.root.winfo_rooty() + 50))
        
        # Main content frame
        content_frame = tk.Frame(result_window, bg=self.bg_color, padx=20, pady=20)
        content_frame.pack(expand=True, fill=tk.BOTH)
        
        if self.ai_vs_ai_mode:
            # AI vs AI mode results
            if winner == 1:
                self.player_score += 1
                self.player_score_label.config(text=f"MINIMAX: {self.player_score}")
                self.status_label.config(text="MINIMAX WINS! 🎉", fg=self.human_color)
                
                result_text = "Minimax AI Wins! 🎉"
                result_color = self.human_color
                
            elif winner == 2:
                self.ai_score += 1
                self.ai_score_label.config(text=f"MCTS: {self.ai_score}")
                self.status_label.config(text="MCTS WINS! 🤖", fg=self.ai_color)
                
                result_text = "MCTS AI Wins! 🤖"
                result_color = self.ai_color
                
            else:
                self.status_label.config(text="IT'S A DRAW! 🤝", fg=self.text_color)
                
                result_text = "It's a Draw!\nBoth AIs played well 🤝"
                result_color = self.text_color
                
            score_text = f"Score: Minimax {self.player_score} - {self.ai_score} MCTS"
            
            # Reset current game metrics for next game
            self.minimax_metrics['current_game_moves'] = 0
            self.minimax_metrics['current_game_time'] = 0
            self.mcts_metrics['current_game_moves'] = 0
            self.mcts_metrics['current_game_time'] = 0
            
            # Format performance metrics for display
            minimax_avg_time = self.minimax_metrics['avg_move_time'] * 1000  # Convert to ms
            mcts_avg_time = self.mcts_metrics['avg_move_time'] * 1000  # Convert to ms
            
            minimax_metrics_text = f"Minimax AI Metrics:\n" \
                                  f"• Total Moves: {self.minimax_metrics['total_moves']}\n" \
                                  f"• Avg Move Time: {minimax_avg_time:.2f} ms\n" \
                                  f"• Min Move Time: {self.minimax_metrics['min_move_time']*1000:.2f} ms\n" \
                                  f"• Max Move Time: {self.minimax_metrics['max_move_time']*1000:.2f} ms"
            
            mcts_metrics_text = f"MCTS AI Metrics:\n" \
                              f"• Total Moves: {self.mcts_metrics['total_moves']}\n" \
                              f"• Avg Move Time: {mcts_avg_time:.2f} ms\n" \
                              f"• Min Move Time: {self.mcts_metrics['min_move_time']*1000:.2f} ms\n" \
                              f"• Max Move Time: {self.mcts_metrics['max_move_time']*1000:.2f} ms"
        else:
            # Human vs AI mode results
            if winner == 1:
                self.player_score += 1
                self.player_score_label.config(text=f"YOU: {self.player_score}")
                self.status_label.config(text="YOU WIN! 🎉", fg=self.human_color)
                
                result_text = "Congratulations!\nYou Win! 🎉"
                result_color = self.human_color
                
            elif winner == 2:
                self.ai_score += 1
                self.ai_score_label.config(text=f"AI: {self.ai_score}")
                self.status_label.config(text="AI WINS! 🤖", fg=self.ai_color)
                
                result_text = "AI Wins!\nBetter luck next time 🤖"
                result_color = self.ai_color
                
            else:
                self.status_label.config(text="IT'S A DRAW! 🤝", fg=self.text_color)
                
                result_text = "It's a Draw!\nWell played 🤝"
                result_color = self.text_color
                
            score_text = f"Score: You {self.player_score} - {self.ai_score} AI"
        
        result_label = tk.Label(content_frame, text=result_text, font=("Arial", 18, "bold"),
                              bg=self.bg_color, fg=result_color, justify=tk.CENTER)
        result_label.pack(pady=10)
        
        score_label = tk.Label(content_frame, text=score_text, font=("Arial", 12),
                             bg=self.bg_color, fg=self.text_color)
        score_label.pack(pady=5)
        
        # Display performance metrics in AI vs AI mode with improved layout
        if self.ai_vs_ai_mode:
            # Create a canvas with scrollbar for metrics
            canvas_frame = tk.Frame(content_frame, bg=self.bg_color)
            canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)
            
            # Add a canvas
            metrics_canvas = tk.Canvas(canvas_frame, bg=self.bg_color, highlightthickness=0)
            metrics_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # Add a scrollbar to the canvas
            scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=metrics_canvas.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Configure the canvas
            metrics_canvas.configure(yscrollcommand=scrollbar.set)
            metrics_canvas.bind('<Configure>', lambda e: metrics_canvas.configure(scrollregion=metrics_canvas.bbox("all")))
            
            # Create a frame inside the canvas
            metrics_frame = tk.Frame(metrics_canvas, bg=self.bg_color)
            metrics_canvas.create_window((0, 0), window=metrics_frame, anchor="nw")
            
            # Add a title for the metrics section
            metrics_title = tk.Label(metrics_frame, text="Performance Comparison", 
                                   font=("Arial", 12, "bold"), bg=self.bg_color, 
                                   fg=self.accent_color, justify=tk.CENTER)
            metrics_title.pack(pady=(0, 10), fill=tk.X)
            
            # Create separate frames for each AI's metrics
            minimax_frame = tk.Frame(metrics_frame, bg=self.bg_color, bd=1, relief=tk.GROOVE,
                                   highlightbackground=self.human_color, highlightthickness=1)
            minimax_frame.pack(pady=5, fill=tk.X, padx=5)
            
            minimax_title = tk.Label(minimax_frame, text="Minimax AI", 
                                   font=("Arial", 11, "bold"), bg=self.bg_color, 
                                   fg=self.human_color)
            minimax_title.pack(pady=(5, 0))
            
            # Split metrics into separate labels for better readability
            tk.Label(minimax_frame, text=f"Total Moves: {self.minimax_metrics['total_moves']}", 
                   font=("Arial", 10), bg=self.bg_color, fg=self.text_color, 
                   justify=tk.LEFT).pack(anchor=tk.W, padx=10)
            
            tk.Label(minimax_frame, text=f"Avg Move Time: {minimax_avg_time:.2f} ms", 
                   font=("Arial", 10), bg=self.bg_color, fg=self.text_color, 
                   justify=tk.LEFT).pack(anchor=tk.W, padx=10)
            
            tk.Label(minimax_frame, text=f"Min Move Time: {self.minimax_metrics['min_move_time']*1000:.2f} ms", 
                   font=("Arial", 10), bg=self.bg_color, fg=self.text_color, 
                   justify=tk.LEFT).pack(anchor=tk.W, padx=10)
            
            tk.Label(minimax_frame, text=f"Max Move Time: {self.minimax_metrics['max_move_time']*1000:.2f} ms", 
                   font=("Arial", 10), bg=self.bg_color, fg=self.text_color, 
                   justify=tk.LEFT).pack(anchor=tk.W, padx=10, pady=(0, 5))
            
            # MCTS metrics frame
            mcts_frame = tk.Frame(metrics_frame, bg=self.bg_color, bd=1, relief=tk.GROOVE,
                               highlightbackground=self.ai_color, highlightthickness=1)
            mcts_frame.pack(pady=5, fill=tk.X, padx=5)
            
            mcts_title = tk.Label(mcts_frame, text="MCTS AI", 
                               font=("Arial", 11, "bold"), bg=self.bg_color, 
                               fg=self.ai_color)
            mcts_title.pack(pady=(5, 0))
            
            tk.Label(mcts_frame, text=f"Total Moves: {self.mcts_metrics['total_moves']}", 
                   font=("Arial", 10), bg=self.bg_color, fg=self.text_color, 
                   justify=tk.LEFT).pack(anchor=tk.W, padx=10)
            
            tk.Label(mcts_frame, text=f"Avg Move Time: {mcts_avg_time:.2f} ms", 
                   font=("Arial", 10), bg=self.bg_color, fg=self.text_color, 
                   justify=tk.LEFT).pack(anchor=tk.W, padx=10)
            
            tk.Label(mcts_frame, text=f"Min Move Time: {self.mcts_metrics['min_move_time']*1000:.2f} ms", 
                   font=("Arial", 10), bg=self.bg_color, fg=self.text_color, 
                   justify=tk.LEFT).pack(anchor=tk.W, padx=10)
            
            tk.Label(mcts_frame, text=f"Max Move Time: {self.mcts_metrics['max_move_time']*1000:.2f} ms", 
                   font=("Arial", 10), bg=self.bg_color, fg=self.text_color, 
                   justify=tk.LEFT).pack(anchor=tk.W, padx=10, pady=(0, 5))
        
        # Close button
        close_button = tk.Button(content_frame, text="Continue", font=("Arial", 12),
                               command=result_window.destroy, bg="#3182ce", fg=self.text_color,
                               activebackground="#2b6cb0", activeforeground=self.text_color,
                               relief=tk.FLAT, padx=15, pady=5, cursor="hand2")
        close_button.pack(pady=10)
    
    def reset_game(self):
        """
        Reset the game with improved error handling
        """
        try:
            # Reset game state
            self.game_logic.reset()
            self.winning_cells = []
            
            # Reset current game performance metrics in AI vs AI mode
            if self.ai_vs_ai_mode:
                self.minimax_metrics['current_game_moves'] = 0
                self.minimax_metrics['current_game_time'] = 0
                self.mcts_metrics['current_game_moves'] = 0
                self.mcts_metrics['current_game_time'] = 0
            
            # Reset board UI with reduced animations to prevent freezing
            try:
                for row in range(self.board.size):
                    for col in range(self.board.size):
                        self.buttons[row][col].config(text="", state=tk.NORMAL, bg=self.btn_color)
                        # Update less frequently to reduce resource usage
                        if (row * self.board.size + col) % 3 == 0:
                            self.root.update()
                            time.sleep(0.005)  # Reduced delay
            except Exception as ui_error:
                print(f"Error resetting UI: {ui_error}")
                # Fallback: reset all buttons at once
                for row in range(self.board.size):
                    for col in range(self.board.size):
                        self.buttons[row][col].config(text="", state=tk.NORMAL, bg=self.btn_color)
            
            # Final UI update
            self.root.update()
            
            if self.ai_vs_ai_mode:
                try:
                    # In AI vs AI mode, always start with player 1 (Minimax AI)
                    self.status_label.config(text="MINIMAX'S TURN", fg=self.human_color)
                    # Use after() instead of direct call to prevent stack overflow
                    self.root.after(800, self.make_ai_move)  # Start AI vs AI game automatically with reduced delay
                except Exception as ai_start_error:
                    print(f"Error starting AI vs AI mode: {ai_start_error}")
            else:
                # Human vs AI mode with improved error handling
                try:
                    # In Human vs AI mode, randomly decide who goes first
                    if random.random() < 0.5:
                        # Human goes first
                        self.game_logic.current_player = 1
                        self.status_label.config(text="YOUR TURN", fg=self.human_color)
                    else:
                        # AI goes first
                        self.game_logic.current_player = 2
                        self.status_label.config(text="AI'S TURN", fg=self.ai_color)
                        # Schedule AI move after a short delay
                        self.root.after(800, self.make_ai_move)
                except Exception as human_ai_error:
                    print(f"Error setting up Human vs AI mode: {human_ai_error}")
                    # Fallback to human first
                    self.game_logic.current_player = 1
                    self.status_label.config(text="YOUR TURN", fg=self.human_color)
        except Exception as e:
            print(f"Critical error in reset_game: {e}")
            # Last resort recovery
            try:
                self.game_logic.reset()
                for row in range(self.board.size):
                    for col in range(self.board.size):
                        self.buttons[row][col].config(text="", state=tk.NORMAL, bg=self.btn_color)
                self.status_label.config(text="GAME RESET", fg=self.text_color)
            except:
                messagebox.showerror("Error", "Failed to reset game. Please restart the application.")