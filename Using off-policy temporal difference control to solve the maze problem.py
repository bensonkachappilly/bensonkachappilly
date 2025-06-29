#%%
author = "Benson Kachappilly"
program_description = "Solving the maze problem by trying to get out of it by using Off-Policy Temporal Difference Control"
method_description = "Using Off-Policy Temporal Difference Control to find the best action at each state by maximizing expected future rewards, learning an optimal policy through exploration and exploitation."

import tkinter as tk
import random

# Create the maze  (0 = open, 1 = wall)
maze = [
    [0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,1,1,0,1,0,1,1,0,0,0,1,1,1,1,1,0],
    [0,1,0,0,1,0,0,1,1,1,0,1,0,1,0,0,0],
    [0,1,1,1,1,1,0,1,1,1,0,0,0,1,0,1,1],
    [0,0,1,0,0,1,0,1,1,1,1,1,1,1,0,1,0],
    [1,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0],
    [0,0,0,1,1,1,0,1,0,0,1,0,1,1,1,1,1],
    [0,1,1,1,0,0,0,1,1,1,1,0,1,1,1,1,1],
    [0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0]
]

#  Start and Exit
start = (0, 0)
goal = (8, 16)

# up, down, left, right
actions = [(-1,0), (1,0), (0,-1), (0,1)]

# Parameters
alpha = 0.5
gamma = 0.9
epsilon = 0.9
episodes = 30
delay = 0.3
# The parameters I played around with to ensure my Q-learning algorithm is working. Firstly, I reduced the episodes to 30 and the epsilon to 0.1 to ensure that the agent is not exploring too much. In doing so I found that the agent was unable to exit and would get stuck in certain parts. This is expected as the agent is not exploring enough and with only 30 episodes, the agent is not able to learn enough. When I increased the epsilon to 0.9 and kept the same number of episodes, the agent was able to explore more and learn faster, while being able to exit the maze. This is expected with Q-learning as it always uses the best future Q-value , hence even random moves give important information for learning the best future action.

# Global variables
Q = {}
cell_size = 50
rows = len(maze)
cols = len(maze[0])
root = None
canvas = None
start_button = None
path = []
agent = None
inter = 0
iterations = 0
quit = False

def build_maze():
    """Function build_maze to build the maze

            Arguments:
             None

            Returns:
             None
    """
    global canvas, start_button
    canvas = tk.Canvas(root, width=cols*cell_size, height=rows*cell_size, bg='white')
    canvas.pack()

    for r in range(rows):
        for c in range(cols):
            color = "white" if maze[r][c] == 0 else "black"
            canvas.create_rectangle(c*cell_size, r*cell_size, (c+1)*cell_size, (r+1)*cell_size, fill=color)

    # Start and Goal
    canvas.create_oval(start[1]*cell_size+10, start[0]*cell_size+10, start[1]*cell_size+30, start[0]*cell_size+30, fill="red")
    canvas.create_oval(goal[1]*cell_size+10, goal[0]*cell_size+10, goal[1]*cell_size+30, goal[0]*cell_size+30, fill="green")

    start_button = tk.Button(root, text="Start", command=start_learning, width=18, height=2, font=("Arial", 18))
    start_button.pack(pady=10)

def valid_moves(state):
    """Function valid_moves to get the valid moves from the current state

            Arguments:
             state -- The current state

            Returns:
             moves -- The valid moves
    """

    moves = []
    # Get the current position
    r, c = state
    # Check all possible actions
    for idx, (dr, dc) in enumerate(actions):
        nr, nc = r + dr, c + dc
        # Check if the new position is within bounds and not a wall
        if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0:
            # Add the move to the list
            moves.append((idx, (nr, nc)))
    return moves

def choose_action(state):
    """Function choose_action to choose the action to take

            Arguments:
             state -- The current state

            Returns:
             action_idx -- The index of the action
             next_state -- The next state
    """
    # Randomly choose an action with probability epsilon
    if random.uniform(0,1) < epsilon:
        moves = valid_moves(state)
        return random.choice(moves)
    else:
        # Choose the action with the highest Q-value
        moves = valid_moves(state)
        qs = [Q.get((state, a_idx), 0) for a_idx, _ in moves]
        # If there are valid moves, choose the one with the highest Q-value
        if qs:
            max_idx = qs.index(max(qs))
            return moves[max_idx]
        # If no valid moves, choose a random move
        else:
            return random.choice(moves)

def q_learning():
    """Function q_learning to implement the Q-learning algorithm

            Arguments:
             None

            Returns:
             None
    """
    # Initialize the Q function
    for ep in range(episodes):
        # Start from the initial state
        state = start
        # Loop until the goal is reached
        while state != goal:
            # Choose an action
            action_idx, next_state = choose_action(state)
            # Get the reward
            reward = 1 if next_state == goal else -0.04
            # Update the Q function
            best_next_q = max([Q.get((next_state, a_idx), 0) for a_idx, _ in valid_moves(next_state)], default=0)
            # Get the old Q value
            old_q = Q.get((state, action_idx), 0)
            # Update the Q value using the Q-learning formula
            Q[(state, action_idx)] = old_q + alpha * (reward + gamma * best_next_q - old_q)
            # Move to the next state
            state = next_state

def get_best_path():
    """Function get_best_path to get the best path from the Q function

            Arguments:
             None

            Returns:
             path -- The best path
    """
    # Initialize parameters
    path = [start]
    state = start
    visited = set()
    # Loop until the goal is reached or no valid moves
    while state != goal and state not in visited:
        visited.add(state)
        moves = valid_moves(state)
        # If no valid moves, break
        if not moves:
            break
        # Get the Q values for the valid moves
        qs = [Q.get((state, a_idx), -float('inf')) for a_idx, _ in moves]
        best_move = moves[qs.index(max(qs))]
        state = best_move[1]
        path.append(state)
    return path

def start_learning():
    """Function start_learning to start the learning process

            Arguments:
             None

            Returns:
             None
    """
    global path, iterations
    print("Author:", author)
    print("Description:", program_description)
    print("Method:", method_description)
    print("Parameters:")
    print(f"  Episodes: {episodes}")
    print(f"  Alpha: {alpha}, Gamma: {gamma}, Epsilon: {epsilon}")

    start_button.config(state="disabled")
    q_learning()
    path = get_best_path()
    iterations = len(path)
    print("Final Path:", path)
    root.after(1000, show_path)

def show_path():
    """Function show_path to show the path taken by the agent

            Arguments:
             None

            Returns:
             None
    """
    global inter, iterations, path, agent
    # Check if the agent has reached the goal or if the program is quitting
    if inter >= iterations:
        return
    # Get the current position of the agent
    i, j = path[inter]
    # Calculate the coordinates of the agent
    x1 = j * cell_size + 10
    y1 = i * cell_size + 10
    x2 = x1 + cell_size - 20
    y2 = y1 + cell_size - 20
    # Draw the agent on the canvas
    if agent:
        canvas.delete(agent)
    agent = canvas.create_oval(x1, y1, x2, y2, fill="blue")
    # Increment the index to show the next position
    inter += 1
    root.after(int(delay*1000), show_path)

def on_closing():
    """Function on_closing to handle the closing of the window

            Arguments:
             None

            Returns:
             None
    """
    global quit
    quit = True
    root.destroy()

def main():
    """Function main to run the program

            Arguments:
             None

            Returns:
             None
    """
    global root
    root = tk.Tk()
    root.title("The Maze - Off-Policy Temporal Difference Control")
    build_maze()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
