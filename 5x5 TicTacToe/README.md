# 🎮 Tic-Tac-Five: Intelligent 5x5 Tic-Tac-Toe with AI Modes

## 🧠 Overview

**Tic-Tac-Five** is a smart twist on the classic game of Tic-Tac-Toe, designed on a **5x5 board** with the goal of forming **four marks in a row** (horizontally, vertically, or diagonally). This project introduces two primary modes:

- 🤖 **AI vs AI Mode** – Minimax (Alpha-Beta Pruning) vs Monte Carlo Tree Search (MCTS)
- 🧍‍♂️ **Human vs AI Mode** – Human plays against Minimax AI in three difficulty levels

Built with Python and Tkinter, this game not only provides an interactive interface but also explores AI algorithms and comparative analysis based on performance metrics.

---

## 📌 Features

### 🧠 AI vs AI Mode

In this fully automated mode:
- Two AIs (Minimax with Alpha-Beta Pruning and MCTS) play against each other.
- After each match, the game displays:
  - ✅ Number of nodes explored
  - 🕒 Average decision time per move
  - 🧠 Memory usage
  - 🏆 Win/Loss/Draw outcomes

This mode helps compare **deterministic vs probabilistic** AI strategies in a structured environment.

---

### 👤 Human vs AI Mode

Play against the AI with 3 difficulty levels:
- 🟢 **Easy** – Random or shallow-depth Minimax
- 🟡 **Medium** – Mid-depth Minimax with heuristic evaluation
- 🔴 **Hard** – Deep Minimax with Alpha-Beta pruning and advanced heuristics

You can choose your difficulty level and test your strategy against a thinking opponent.

---

## 🧩 Technologies Used

- **Python 3.x**
- **Tkinter** (GUI)
- **Minimax Algorithm with Alpha-Beta Pruning**
- **Monte Carlo Tree Search (MCTS)**

---

## 📊 Performance Metrics (AI vs AI)

After each AI vs AI game, the following stats are logged:
- 🔍 **Nodes Explored** – Shows search space complexity
- ⏱ **Decision Time** – Average thinking time per move
- 📦 **Memory Usage** – Resource efficiency
- ⚔️ **Results** – Number of wins/draws for each algorithm

These are useful for AI benchmarking and academic analysis.

---

## 🎨 User Interface

- Intuitive 5x5 grid with clickable buttons
- Mode and difficulty selection menus
- Real-time game updates and visual feedback
- Restart and exit options with confirmation dialogs
- Scoreboard for tracking wins/losses

---

## 📈 Complexity & Analysis

| Algorithm        | Time Complexity            | Space Complexity         |
|------------------|-----------------------------|---------------------------|
| Minimax          | O(b^d)                      | O(d)                      |
| Minimax + Alpha-Beta | O(b^(d/2))               | O(d)                      |
| MCTS             | O(n log n) for `n` simulations | Depends on tree size     |

> `b`: branching factor, `d`: depth of search

Comparisons are plotted in graphs and tables for better visualization.

---

## 📁 Folder Structure


# **Project Report: 5x5 Tic-Tac-Toe with Minimax AI**

**Submitted By:**  
- Muhammad Abbas (23k-0068)  
- Hamza Sheikh (23k-0060)  
- Muhammad Sabeeh (23k-0002)  
- Zaid Bin Naveed (23k-0028)  

**Course:** Artificial Intelligence  
**Instructor:** Miss Ramsha Jut  
**Submission Date:** August 5, 2025  

---

## **Executive Summary**  
This project involves the development of a modernized 5x5 Tic-Tac-Toe game featuring an AI opponent. The primary objective was to enhance the traditional 3x3 Tic-Tac-Toe by expanding the board size to 5x5 and introducing a win condition of five consecutive marks. The AI leverages the Minimax algorithm with Alpha-Beta pruning to provide a challenging and adaptive opponent. Multiple difficulty levels (easy, medium, hard) were implemented to cater to players of varying skill levels. The game also boasts a sleek graphical user interface (GUI) with animations and a dark theme for an engaging user experience.

---

## **1. Introduction**  

### **Background**  
Tic-Tac-Toe is a classic two-player game traditionally played on a 3x3 grid. While simple, it serves as an excellent foundation for exploring game theory and AI algorithms. This project extends the game to a 5x5 grid, significantly increasing its complexity and strategic depth. The inclusion of an AI opponent allows for single-player gameplay, making the game more versatile and enjoyable.  

### **Objectives**  
The key objectives of this project were:  
1. **Game Development:** Create a 5x5 Tic-Tac-Toe game with a graphical user interface.  
2. **AI Implementation:** Develop an AI opponent using the Minimax algorithm enhanced with Alpha-Beta pruning for efficient decision-making.  
3. **Difficulty Customization:** Introduce multiple difficulty levels to accommodate players of all skill levels.  
4. **User Experience:** Enhance gameplay with modern UI design, animations, and intuitive controls.  

---

## **2. Game Description**  

### **Original Game Rules**  
The traditional Tic-Tac-Toe game involves two players taking turns to place their marks (X or O) on a 3x3 grid. The first player to align three marks horizontally, vertically, or diagonally wins.  

### **Innovations and Modifications**  
This project introduces the following innovations:  
- **Expanded Board:** A 5x5 grid replaces the traditional 3x3 grid, requiring players to align five consecutive marks to win.  
- **Advanced AI:** The AI opponent uses the Minimax algorithm with Alpha-Beta pruning to make strategic moves.  
- **Difficulty Levels:** Players can choose from easy, medium, or hard difficulty levels, each altering the AI's decision-making process.  
- **Modern UI:** The game features a dark-themed interface with smooth animations for a visually appealing experience.  

---

## **3. AI Approach and Methodology**  

### **AI Techniques Used**  
The AI employs the **Minimax algorithm**, a decision-making strategy used in two-player games to minimize potential loss and maximize gain. To optimize performance, **Alpha-Beta pruning** was integrated, reducing the number of nodes evaluated by the algorithm without affecting the final decision.  

### **Algorithm and Heuristic Design**  
The AI evaluates the game board by analyzing windows of five cells. Each potential move is scored based on:  
- Opportunities to create a winning sequence.  
- Threats posed by the opponent that need to be blocked.  

**Difficulty Levels:**  
- **Easy:** The AI frequently makes suboptimal moves, providing a less challenging experience.  
- **Medium:** The AI occasionally makes suboptimal moves, balancing challenge and playability.  
- **Hard:** The AI always selects the best possible move, offering a formidable opponent.  

### **AI Performance Evaluation**  
The AI's performance was rigorously tested across all difficulty levels:  
- **Hard Mode:** The AI consistently makes optimal moves, ensuring a high level of challenge.  
- **Medium Mode:** The AI strikes a balance between challenge and accessibility.  
- **Easy Mode:** The AI's suboptimal moves make it suitable for beginners.  

The Alpha-Beta pruning significantly improved the algorithm's efficiency, enabling quick decision-making even on the larger 5x5 grid.  

---

## **4. Game Mechanics and Rules**  

### **Modified Game Rules**  
- **Board Size:** 5x5 grid.  
- **Win Condition:** Align five consecutive marks horizontally, vertically, or diagonally.  
- **Turn-Based Play:** Players alternate turns until a win or draw is achieved.  

### **Winning Conditions**  
- The first player to align five marks wins.  
- If the board fills without a winner, the game ends in a draw.  

---

## **5. Implementation and Development**  

### **Development Process**  
The game was developed using **Python** and the **tkinter** library for the GUI. The AI logic was implemented from scratch, incorporating Minimax and Alpha-Beta pruning. Key development stages included:  
1. **Prototyping:** Initial design of the game mechanics and board.  
2. **AI Integration:** Implementing and refining the Minimax algorithm.  
3. **UI Design:** Creating a modern, user-friendly interface with animations.  
4. **Testing:** Evaluating AI performance and gameplay balance.  

### **Programming Languages and Tools**  
- **Language:** Python  
- **Libraries:** tkinter (GUI), standard Python libraries (for logic and animations).  
- **Tools:** Python IDEs (e.g., PyCharm, VS Code).  

### **Challenges Encountered**  
1. **Balancing AI Difficulty:** Adjusting the AI's decision-making to ensure fairness across difficulty levels.  
2. **UI Animations:** Implementing smooth animations in tkinter required creative solutions due to its limited native support.  
3. **Performance Optimization:** Ensuring the Minimax algorithm remained efficient on the larger 5x5 grid.  

---

## **6. Team Contributions**  

| Team Member          | Responsibilities                                                                 |
|----------------------|---------------------------------------------------------------------------------|
| **Hamza Sheikh**     | Developed the AI algorithms (Minimax, Alpha-Beta Pruning).                      |
| **Muhammad Sabeeh**  | Designed the modified game rules and board layout.                              |
| **Muhammad Abbas**   | Implemented the user interface and integrated the AI with gameplay.             |
| **Zaid Bin Naveed**  | Conducted performance testing and evaluated the AI's decision-making.           |

---

## **7. Results and Discussion**  
The project successfully achieved its objectives:  
- The 5x5 Tic-Tac-Toe game offers a fresh take on the classic, with increased strategic depth.  
- The AI provides a dynamic challenge, adaptable to player skill levels.  
- The modern UI enhances the overall user experience.  

Future improvements could include:  
- Additional customization options (e.g., adjustable board sizes).  
- Enhanced animations and sound effects.  
- Multiplayer support for online play.  

---

## **8. References**  
- Minimax Algorithm and Alpha-Beta Pruning (AI literature).  
- Python tkinter documentation (GUI development).  
- Online resources on Tic-Tac-Toe game implementations.  

--- 



https://github.com/user-attachments/assets/1e1bf3e3-db29-421f-8380-4d6d34649b85



