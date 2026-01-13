import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pickle #this is for the PKL file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the saved model
with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict Price Function
def predict_price():
    try:
        inputs = [entry.get().strip() for entry in entries]

        # Ensure valid input: Check for any missing or invalid inputs
        if not all(inputs):
            raise ValueError("All fields are required!")

        # Convert inputs to float (for numerical values)
        inputs_transformed = []
        for idx, col in enumerate(columns):
            try:
                if col in datatransformed:  # Categorical columns
                    inputs_transformed.append(inputs[idx])
                else:  # Numerical columns
                    inputs_transformed.append(float(inputs[idx]))
            except ValueError:
                raise ValueError(f"Invalid value for {col}. Please enter a number.")

        # Convert inputs into a DataFrame
        input_df = pd.DataFrame([inputs_transformed], columns=columns)

        # Make prediction
        prediction = model.predict(input_df)[0]

        # Display the result
        result_label.config(text=f"Predicted Price: Rs {prediction:,.2f}")
    except ValueError as e:
        messagebox.showerror("Input Error", str(e))
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

# Reset Input Fields
def reset_fields():
    for entry in entries:
        entry.delete(0, tk.END)  # Clear each input field
    result_label.config(text="")  # Clear the result label

# Gradient Background Function
def create_gradient(canvas, color1, color2, width, height):
    """Draws a vertical gradient from `color1` to `color2`."""
    r1, g1, b1 = app.winfo_rgb(color1)
    r2, g2, b2 = app.winfo_rgb(color2)

    r_ratio = (r2 - r1) / height
    g_ratio = (g2 - g1) / height
    b_ratio = (b2 - b1) / height

    for i in range(height):
        nr = int(r1 + (r_ratio * i))
        ng = int(g1 + (g_ratio * i))
        nb = int(b1 + (b_ratio * i))
        color = f'#{nr:04x}{ng:04x}{nb:04x}'
        canvas.create_line(0, i, width, i, fill=color)

# Graphs Display Function
def show_graphs():
    # Create a new window for graphs
    graph_window = tk.Toplevel(app)
    graph_window.title("Graphs")
    graph_window.geometry("800x600")
    
    # Load dataset
    data = pd.read_csv("Housing.csv")
    # Encode categorical variables ('yes'/'no' to 1/0)
    categorical_columns = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea", "furnishingstatus"]
    for col in categorical_columns:
        data[col] = data[col].map({'yes': 1, 'no': 0})

    # Create a figure and axes for the graphs
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))  # 2x2 grid of subplots
    
    # Generate some of the plots from model.py
    axes[0, 0].hist(data["bedrooms"], bins=30, color='skyblue', edgecolor='black')
    axes[0, 0].set_title("Distribution of Bedrooms")
    
    axes[0, 1].boxplot(data["price"])
    axes[0, 1].set_title("Boxplot of Price")
    
    axes[1, 0].scatter(data["area"], data["price"], alpha=0.5)
    axes[1, 0].set_title("Price vs. Area")
    
    # Calculate correlation matrix (excluding non-numeric columns)
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=axes[1, 1])
    axes[1, 1].set_title("Correlation Heatmap")

    # Use FigureCanvasTkAgg to embed the matplotlib figure into the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=graph_window)  # A tk.DrawingArea.
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Tooltip Class for Help
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self.show_tooltip)
        widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        if self.tip_window:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, background="#FFFFE0", relief="solid", borderwidth=1, font=("Helvetica", 10))
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None

# Load columns from dataset
columns = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea", 
           "furnishingstatus", "area", "bedrooms", "bathrooms", "stories", "parking"]
datatransformed = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", 
                   "prefarea", "furnishingstatus"]  # Categorical columns

# Create Application Window
app = tk.Tk()
app.title("House Price Predictor")
app.geometry("600x800")
app.resizable(True, True)

# Create a Canvas for Background Gradient
canvas = tk.Canvas(app, width=600, height=800)
canvas.pack(fill="both", expand=True)

# Apply Gradient Background (light blue to pale green)
create_gradient(canvas, "#c31432", "#240b36", 2000, 800)  # Blue to dark gradient

# Add a Black Border Frame
outer_frame = tk.Frame(canvas, relief="solid", borderwidth=5, bg="black")
outer_frame.place(relx=0.05, rely=0.05, relwidth=0.9, relheight=0.9)  # Adjusted to fit all components

# Create a Frame Inside the Border for Content
main_frame = ttk.Frame(outer_frame, padding="20")
main_frame.pack(fill="both", expand=True)

# Add Header Label
header_label = ttk.Label(main_frame, text="House Price Predictor", font=("Helvetica", 16, "bold"))
header_label.pack(pady=10)

# Input Fields with Border
border_frame = ttk.Frame(main_frame, padding="10", relief="solid", borderwidth=2)
border_frame.pack(pady=20)

entries = []
tooltips = [
    "Enter 'yes' or 'no' for mainroad.",
    "Enter 'yes' or 'no' for guestroom.",
    "Enter 'yes' or 'no' for basement.",
    "Enter 'yes' or 'no' for hotwaterheating.",
    "Enter 'yes' or 'no' for airconditioning.",
    "Enter 'yes' or 'no' for prefarea.",
    "Enter 'furnished', 'semi-furnished', or 'unfurnished'.",
    "Enter the area in square feet.",
    "Enter the number of bedrooms.",
    "Enter the number of bathrooms.",
    "Enter the number of stories.",
    "Enter the parking capacity."
]

for idx, (col, tip) in enumerate(zip(columns, tooltips), start=1):
    label = ttk.Label(border_frame, text=f"{col.capitalize()}:", font=("Helvetica", 12))
    label.grid(row=idx, column=0, padx=10, pady=5, sticky=tk.E)
    entry = ttk.Entry(border_frame, font=("Helvetica", 12), width=25)
    entry.grid(row=idx, column=1, padx=10, pady=5)
    entries.append(entry)
    ToolTip(entry, tip)

# Predict Button
predict_button = ttk.Button(main_frame, text="Predict Price", command=predict_price)
predict_button.pack(pady=2)

# Reset Button
reset_button = ttk.Button(main_frame, text="Reset Fields", command=reset_fields)
reset_button.pack(pady=2)

# Show Graphs Button
show_graphs_button = ttk.Button(main_frame, text="Show Graphs", command=show_graphs)
show_graphs_button.pack(pady=2)

# Result Label
result_label = ttk.Label(main_frame, text="", font=("Helvetica", 14, "bold"))
result_label.pack(pady=2)

# Footer
footer_label = ttk.Label(main_frame, text="Developed by: Muhammad, Hamza, Sabeeh", font=("Helvetica", 10))
footer_label.pack(pady=0)

app.mainloop()
