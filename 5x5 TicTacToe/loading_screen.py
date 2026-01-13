import tkinter as tk
import time
import random
from tkinter import font as tkfont
import math

class LoadingScreen:
    def __init__(self, root, callback):
        self.root = root
        self.callback = callback  # Function to call when loading is complete
        
        # Modern color scheme
        self.bg_color = '#0f0f1a'  # Deep blue-black
        self.accent_color = '#6c5ce7'  # Vibrant purple
        self.accent_secondary = '#00cec9'  # Teal
        self.text_color = '#ffffff'  # White
        self.progress_bg = '#2d3436'  # Dark gray
        
        # Configure the window
        self.root.configure(bg=self.bg_color)
        self.root.title("Tic Tac Toe AI")
        
        # Center the window on screen
        window_width = 600
        window_height = 400
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        # Fonts
        self.title_font = tkfont.Font(family="Helvetica", size=32, weight="bold")
        self.subtitle_font = tkfont.Font(family="Helvetica", size=14)
        
        # Create main frame
        self.main_frame = tk.Frame(self.root, bg=self.bg_color)
        self.main_frame.pack(expand=True, fill=tk.BOTH, padx=40, pady=40)
        
        # Title with gradient effect
        self.title_label = tk.Label(self.main_frame, text="TIC TAC TOE", 
                                  font=self.title_font, bg=self.bg_color, fg=self.accent_color)
        self.title_label.pack(pady=(20, 10))
        
        # Subtitle
        self.subtitle_label = tk.Label(self.main_frame, text="POWERED BY AI", 
                                     font=self.subtitle_font, bg=self.bg_color, fg=self.accent_secondary)
        self.subtitle_label.pack(pady=(0, 40))
        
        # Canvas for loading animation
        self.canvas_size = 200
        self.canvas = tk.Canvas(self.main_frame, width=self.canvas_size, height=self.canvas_size, 
                              bg=self.bg_color, highlightthickness=0)
        self.canvas.pack(pady=20)
        
        # Progress bar frame
        self.progress_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        self.progress_frame.pack(fill=tk.X, pady=30)
        
        # Progress bar
        self.progress_bar_width = 400
        self.progress_bar_height = 8
        self.progress_bar = tk.Canvas(self.progress_frame, width=self.progress_bar_width, 
                                    height=self.progress_bar_height, bg=self.progress_bg,
                                    highlightthickness=0)
        self.progress_bar.pack()
        
        # Progress text
        self.progress_text = tk.Label(self.progress_frame, text="Loading...", 
                                    font=("Helvetica", 10), bg=self.bg_color, fg=self.text_color)
        self.progress_text.pack(pady=(10, 0))
        
        # Initialize loading animation
        self.progress = 0
        self.loading_speed = 0.5  # Controls how fast the loading progresses
        self.particles = []
        self.create_particles()
        self.animate_loading()
        
    def create_particles(self):
        """Create particles for the loading animation"""
        self.particles = []
        center_x = self.canvas_size / 2
        center_y = self.canvas_size / 2
        radius = 70
        
        # Create X and O particles
        for i in range(15):
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, radius)
            x = center_x + distance * math.cos(angle)
            y = center_y + distance * math.sin(angle)
            size = random.uniform(5, 15)
            speed = random.uniform(0.5, 2)
            symbol = random.choice(['X', 'O'])
            color = self.accent_color if symbol == 'X' else self.accent_secondary
            
            particle = {
                'x': x,
                'y': y,
                'size': size,
                'speed': speed,
                'angle': angle,
                'symbol': symbol,
                'color': color,
                'opacity': random.uniform(0.3, 1.0),
                'distance': distance
            }
            self.particles.append(particle)
    
    def animate_particles(self):
        """Animate the particles"""
        self.canvas.delete("all")
        center_x = self.canvas_size / 2
        center_y = self.canvas_size / 2
        
        # Draw a pulsing circle
        pulse = (math.sin(time.time() * 3) + 1) / 2  # Value between 0 and 1
        radius = 70 + pulse * 10
        
        # Draw outer glow
        for i in range(10):
            alpha = 0.1 - (i * 0.01)
            r = radius + i * 2
            self.canvas.create_oval(center_x - r, center_y - r, center_x + r, center_y + r, 
                                   outline=self.accent_color, width=1, fill='')
        
        # Update and draw particles
        for particle in self.particles:
            # Update position with circular motion
            particle['angle'] += 0.02 * particle['speed']
            particle['distance'] = min(70, particle['distance'] + 0.1 * (random.random() - 0.5))
            
            particle['x'] = center_x + particle['distance'] * math.cos(particle['angle'])
            particle['y'] = center_y + particle['distance'] * math.sin(particle['angle'])
            
            # Randomly change opacity for twinkling effect
            particle['opacity'] = min(1.0, max(0.3, particle['opacity'] + 0.05 * (random.random() - 0.5)))
            
            # Draw the particle
            if particle['symbol'] == 'X':
                size = particle['size']
                x, y = particle['x'], particle['y']
                self.canvas.create_line(x-size/2, y-size/2, x+size/2, y+size/2, 
                                       fill=particle['color'], width=2)
                self.canvas.create_line(x+size/2, y-size/2, x-size/2, y+size/2, 
                                       fill=particle['color'], width=2)
            else:  # O
                size = particle['size']
                x, y = particle['x'], particle['y']
                self.canvas.create_oval(x-size/2, y-size/2, x+size/2, y+size/2, 
                                       outline=particle['color'], width=2)
    
    def update_progress_bar(self):
        """Update the progress bar"""
        self.progress_bar.delete("all")
        width = (self.progress / 100) * self.progress_bar_width
        
        # Create gradient effect
        for i in range(int(width)):
            # Calculate color gradient from accent_secondary to accent_color
            ratio = i / self.progress_bar_width
            r1, g1, b1 = int(self.accent_secondary[1:3], 16), int(self.accent_secondary[3:5], 16), int(self.accent_secondary[5:7], 16)
            r2, g2, b2 = int(self.accent_color[1:3], 16), int(self.accent_color[3:5], 16), int(self.accent_color[5:7], 16)
            r = int(r1 + (r2 - r1) * ratio)
            g = int(g1 + (g2 - g1) * ratio)
            b = int(b1 + (b2 - b1) * ratio)
            color = f'#{r:02x}{g:02x}{b:02x}'
            
            self.progress_bar.create_line(i, 0, i, self.progress_bar_height, fill=color)
        
        # Add glow effect at the end of the progress bar
        if width > 5:
            glow_size = 15
            self.progress_bar.create_oval(width-glow_size, -glow_size/2, width+glow_size, self.progress_bar_height+glow_size/2, 
                                         fill=self.accent_color, outline='')
    
    def animate_loading(self):
        """Animate the loading screen"""
        # Animate particles
        self.animate_particles()
        
        # Update progress
        if self.progress < 100:
            # Simulate loading with random increments
            increment = random.uniform(0.2, 1.0) * self.loading_speed
            self.progress = min(100, self.progress + increment)
            
            # Update progress bar
            self.update_progress_bar()
            
            # Update loading text
            loading_texts = ["Loading assets", "Initializing AI", "Preparing game board", "Almost ready"]
            text_index = min(3, int(self.progress / 25))
            dots = "." * (int(time.time() * 2) % 4)
            self.progress_text.config(text=f"{loading_texts[text_index]}{dots}")
            
            # Schedule next animation frame
            self.root.after(30, self.animate_loading)
        else:
            # Loading complete
            self.progress_text.config(text="Ready!")
            self.root.after(500, self.finish_loading)
    
    def finish_loading(self):
        """Finish the loading process and transition to the next screen"""
        # Fade out effect
        self.fade_out()
    
    def fade_out(self, alpha=1.0):
        """Create a fade out effect"""
        if alpha > 0:
            alpha -= 0.05
            self.main_frame.configure(bg=self.bg_color)
            # Fade out all widgets
            for widget in self.main_frame.winfo_children():
                if isinstance(widget, tk.Label):
                    current_fg = widget.cget("fg")
                    r, g, b = self.root.winfo_rgb(current_fg)
                    r, g, b = r//256, g//256, b//256
                    new_color = f'#{int(r*alpha):02x}{int(g*alpha):02x}{int(b*alpha):02x}'
                    widget.configure(fg=new_color)
            
            self.root.after(30, lambda: self.fade_out(alpha))
        else:
            # Destroy loading screen widgets and call the callback
            for widget in self.main_frame.winfo_children():
                widget.destroy()
            self.main_frame.destroy()
            self.callback()