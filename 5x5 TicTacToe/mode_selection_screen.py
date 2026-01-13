import tkinter as tk
import time
from tkinter import font as tkfont

class ModeSelectionScreen:
    def __init__(self, root, callback):
        self.root = root
        self.callback = callback  # Function to call with selected mode
        
        # Modern color scheme (matching the existing screens)
        self.bg_color = '#0f0f1a'  # Deep blue-black
        self.accent_color = '#6c5ce7'  # Vibrant purple
        self.accent_secondary = '#00cec9'  # Teal
        self.text_color = '#ffffff'  # White
        self.human_ai_color = '#4caf50'  # Green
        self.ai_ai_color = '#ff9800'  # Orange
        
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
        self.title_label = tk.Label(self.main_frame, text="SELECT GAME MODE", 
                                  font=self.title_font, bg=self.bg_color, fg=self.accent_color)
        self.title_label.pack(pady=(20, 10))
        
        # Subtitle
        self.subtitle_label = tk.Label(self.main_frame, text="Choose how you want to play the game", 
                                     font=self.subtitle_font, bg=self.bg_color, fg=self.text_color)
        self.subtitle_label.pack(pady=(0, 40))
        
        # Mode buttons frame
        self.buttons_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        self.buttons_frame.pack(pady=20)
        
        # Create mode selection buttons with hover effects
        self.create_mode_buttons()
        
        # Start with animation
        self.animate_entrance()
    
    def create_mode_buttons(self):
        """Create stylish mode selection buttons"""
        modes = [
            {"name": "HUMAN VS AI", "color": self.human_ai_color, "hover": "#66bb6a", "value": "human_vs_ai",
             "description": "Play against the AI with different difficulty levels"},
            {"name": "AI VS AI", "color": self.ai_ai_color, "hover": "#ffa726", "value": "ai_vs_ai",
             "description": "Watch two different AI algorithms compete against each other"}
        ]
        
        self.mode_buttons = []
        
        for i, mode in enumerate(modes):
            # Create button frame for each mode with padding
            button_frame = tk.Frame(self.buttons_frame, bg=self.bg_color, padx=15, pady=15)
            button_frame.grid(row=0, column=i, padx=25)
            
            # Button with rounded corners effect
            button = tk.Button(button_frame, text=mode["name"], font=self.button_font,
                             bg=mode["color"], fg=self.text_color, width=12, height=2,
                             relief=tk.FLAT, cursor="hand2",
                             command=lambda m=mode: self.select_mode(m))
            button.pack(pady=(0, 5))
            
            # Description label
            desc_label = tk.Label(button_frame, text=mode["description"],
                                font=("Helvetica", 10), bg=self.bg_color, fg=self.text_color,
                                wraplength=180)
            desc_label.pack()
            
            # Store button reference and add hover effects
            self.mode_buttons.append((button, mode))
            
            # Add hover effects
            button.bind("<Enter>", lambda event, b=button, h=mode["hover"]: self.on_button_hover(b, h))
            button.bind("<Leave>", lambda event, b=button, c=mode["color"]: self.on_button_leave(b, c))
    
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
        """Animate the entrance of the mode selection screen"""
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
    
    def select_mode(self, mode):
        """Handle mode selection with animation"""
        # Flash effect on selected button
        button = next(b for b, m in self.mode_buttons if m["name"] == mode["name"])
        
        # Animate selection
        for _ in range(3):
            button.config(bg=self.accent_color)
            self.root.update()
            time.sleep(0.05)
            button.config(bg=mode["color"])
            self.root.update()
            time.sleep(0.05)
        
        # Fade out effect
        self.fade_out(mode)
    
    def fade_out(self, selected_mode, alpha=1.0):
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
            
            self.root.after(30, lambda: self.fade_out(selected_mode, alpha))
        else:
            # Destroy mode selection screen widgets and call the callback with selected mode
            for widget in self.main_frame.winfo_children():
                widget.destroy()
            self.main_frame.destroy()
            
            # Call callback with mode info
            self.callback(selected_mode["value"])
