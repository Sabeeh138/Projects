import tkinter as tk
import time
import random
from tkinter import font as tkfont

class DifficultyScreen:
    def __init__(self, root, callback):
        self.root = root
        self.callback = callback  # Function to call with selected difficulty
        
        # Modern color scheme
        self.bg_color = '#0f0f1a'  # Deep blue-black
        self.accent_color = '#6c5ce7'  # Vibrant purple
        self.accent_secondary = '#00cec9'  # Teal
        self.text_color = '#ffffff'  # White
        self.easy_color = '#4caf50'  # Green
        self.medium_color = '#ff9800'  # Orange
        self.hard_color = '#f44336'  # Red
        
        # Configure the window
        self.root.configure(bg=self.bg_color)
        
        # Center the window on screen
        window_width = 600
        window_height = 400
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        # Fonts
        self.title_font = tkfont.Font(family="Helvetica", size=28, weight="bold")
        self.subtitle_font = tkfont.Font(family="Helvetica", size=14)
        self.button_font = tkfont.Font(family="Helvetica", size=16, weight="bold")
        
        # Create main frame with padding
        self.main_frame = tk.Frame(self.root, bg=self.bg_color)
        self.main_frame.pack(expand=True, fill=tk.BOTH, padx=40, pady=40)
        
        # Title
        self.title_label = tk.Label(self.main_frame, text="SELECT DIFFICULTY", 
                                  font=self.title_font, bg=self.bg_color, fg=self.accent_color)
        self.title_label.pack(pady=(20, 10))
        
        # Subtitle
        self.subtitle_label = tk.Label(self.main_frame, text="Choose how challenging you want the AI to be", 
                                     font=self.subtitle_font, bg=self.bg_color, fg=self.text_color)
        self.subtitle_label.pack(pady=(0, 40))
        
        # Difficulty buttons frame
        self.buttons_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        self.buttons_frame.pack(pady=20)
        
        # Create difficulty buttons with hover effects
        self.create_difficulty_buttons()
        
        # Start with animation
        self.animate_entrance()
    
    def create_difficulty_buttons(self):
        """Create stylish difficulty selection buttons"""
        difficulties = [
            {"name": "EASY", "color": self.easy_color, "hover": "#66bb6a", "value": "easy", "depth": 1,
             "description": "For casual play and beginners"},
            {"name": "MEDIUM", "color": self.medium_color, "hover": "#ffa726", "value": "medium", "depth": 3,
             "description": "Balanced challenge for most players"},
            {"name": "HARD", "color": self.hard_color, "hover": "#ef5350", "value": "hard", "depth": 5,
             "description": "For experienced players seeking a challenge"}
        ]
        
        self.difficulty_buttons = []
        
        for i, diff in enumerate(difficulties):
            # Create button frame for each difficulty with padding
            button_frame = tk.Frame(self.buttons_frame, bg=self.bg_color, padx=15, pady=15)
            button_frame.grid(row=0, column=i, padx=15)
            
            # Button with rounded corners effect
            button = tk.Button(button_frame, text=diff["name"], font=self.button_font,
                             bg=diff["color"], fg=self.text_color, width=10, height=2,
                             relief=tk.FLAT, cursor="hand2",
                             command=lambda d=diff: self.select_difficulty(d))
            button.pack(pady=(0, 5))
            
            # Description label
            desc_label = tk.Label(button_frame, text=diff["description"],
                                font=("Helvetica", 10), bg=self.bg_color, fg=self.text_color,
                                wraplength=150)
            desc_label.pack()
            
            # Store button reference and add hover effects
            self.difficulty_buttons.append((button, diff))
            
            # Add hover effects
            button.bind("<Enter>", lambda event, b=button, h=diff["hover"]: self.on_button_hover(b, h))
            button.bind("<Leave>", lambda event, b=button, c=diff["color"]: self.on_button_leave(b, c))
    
    def on_button_hover(self, button, hover_color):
        """Handle button hover effect"""
        button.config(bg=hover_color)
        # Add slight grow effect
        button.config(padx=2, pady=2)
    
    def on_button_leave(self, button, original_color):
        """Handle button leave effect"""
        button.config(bg=original_color)
        # Remove grow effect
        button.config(padx=0, pady=0)
    
    def animate_entrance(self):
        """Animate the entrance of the difficulty selection screen"""
        # Hide all widgets initially
        self.title_label.pack_forget()
        self.subtitle_label.pack_forget()
        self.buttons_frame.pack_forget()
        
        # Fade in title
        self.root.after(200, lambda: self.fade_in_widget(self.title_label, 0))
        
        # Fade in subtitle after title
        self.root.after(800, lambda: self.fade_in_widget(self.subtitle_label, 0))
        
        # Fade in buttons after subtitle
        self.root.after(1400, lambda: self.fade_in_widget(self.buttons_frame, 0))
    
    def fade_in_widget(self, widget, alpha=0):
        """Fade in a widget gradually"""
        if alpha < 1:
            # Pack the widget if it's not already visible
            if widget == self.title_label:
                widget.pack(pady=(20, 10))
            elif widget == self.subtitle_label:
                widget.pack(pady=(0, 40))
            elif widget == self.buttons_frame:
                widget.pack(pady=20)
            
            # Increase alpha for fade-in effect
            alpha += 0.1
            
            # Apply alpha to text color
            if widget in [self.title_label, self.subtitle_label]:
                if widget == self.title_label:
                    base_color = self.accent_color
                else:
                    base_color = self.text_color
                    
                r, g, b = self.root.winfo_rgb(base_color)
                r, g, b = r//256, g//256, b//256
                new_color = f'#{int(r*alpha):02x}{int(g*alpha):02x}{int(b*alpha):02x}'
                widget.configure(fg=new_color)
            
            # Schedule next fade step
            self.root.after(30, lambda: self.fade_in_widget(widget, alpha))
    
    def select_difficulty(self, difficulty):
        """Handle difficulty selection with animation"""
        # Flash effect on selected button
        button = next(b for b, d in self.difficulty_buttons if d["name"] == difficulty["name"])
        
        # Animate selection
        for _ in range(3):
            button.config(bg=self.accent_color)
            self.root.update()
            time.sleep(0.05)
            button.config(bg=difficulty["color"])
            self.root.update()
            time.sleep(0.05)
        
        # Fade out effect
        self.fade_out(difficulty)
    
    def fade_out(self, selected_difficulty, alpha=1.0):
        """Create a fade out effect before transitioning"""
        if alpha > 0:
            alpha -= 0.1
            # Fade out all widgets
            for widget in self.main_frame.winfo_children():
                if isinstance(widget, tk.Label):
                    current_fg = widget.cget("fg")
                    r, g, b = self.root.winfo_rgb(current_fg)
                    r, g, b = r//256, g//256, b//256
                    new_color = f'#{int(r*alpha):02x}{int(g*alpha):02x}{int(b*alpha):02x}'
                    widget.configure(fg=new_color)
            
            self.root.after(30, lambda: self.fade_out(selected_difficulty, alpha))
        else:
            # Destroy difficulty screen widgets and call the callback with selected difficulty
            for widget in self.main_frame.winfo_children():
                widget.destroy()
            self.main_frame.destroy()
            
            # Call callback with difficulty info
            self.callback(selected_difficulty["value"], selected_difficulty["depth"])
