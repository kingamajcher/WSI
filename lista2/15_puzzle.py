import random
from typing import List, Tuple, Callable
from heapdict import heapdict
import time
from heuristics import Heuristics
from datetime import datetime

size = 4

def generate_puzzle(size: int) -> Tuple[int]:
    """
    Generating a solvable puzzle of given size. Number 0 represents the empty tile"""

    numbers = list(range(1, size * size))

    while True:
        random.shuffle(numbers)
        puzzle = numbers + [0]

        if is_solvable(puzzle):
            return tuple(puzzle)
        
def generate_solved_puzzle(size: int) -> Tuple[int]:
    """
    Generating a solved puzzle of given size. Number 0 represents the empty tile
    """
    return tuple(list(range(1, size * size)) + [0])

def generate_puzzle_x_states_ago(x: int) -> Tuple[int]:
    """
    Generating a puzzle x states ago from the solved state. Number 0 represents the empty tile
    """
    solved_puzle = generate_solved_puzzle(size)
    puzzle = list(solved_puzle)

    for _ in range(x):
        zero_index = puzzle.index(0)
        zero_row = zero_index // size
        zero_col = zero_index % size

        new_row, new_col = move_tile_randomly(zero_row, zero_col)

        puzzle[zero_index], puzzle[new_row * size + new_col] = puzzle[new_row * size + new_col], puzzle[zero_index]
    
    return tuple(puzzle)


def is_solvable(puzzle: List[int]) -> bool:
    """
    Assuming the blank tile is always on the last place in the last row, a puzzle is solvable if number of inversions is even.\
    """
    inversions = 0

    for x in range(size * size - 1):
        for y in range(x + 1, size * size):
            if puzzle[x] and puzzle[y] and puzzle[x] > puzzle[y]:
                inversions += 1

    return inversions % 2 == 0 

def move_tile_randomly(row: int, col: int) -> Tuple[int, int]:
    """
    Moving tile in random valid direction
    """
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while True:
        choosen_direction = random.choice(directions)
        new_row = row + choosen_direction[0]
        new_col = col + choosen_direction[1]

        if 0 <= new_row < size and 0 <= new_col < size:
            return new_row, new_col
        
def calculate_neighboring_states(current_state: Tuple[int]) -> List[Tuple[int]]:
    """
    Calculate all neighboring states of the current state
    """
    neighbors = []
    zero_index = current_state.index(0)
    zero_row = zero_index // size
    zero_col = zero_index % size

    for move, direction in [("RIGHT", (0, 1)), ("DOWN", (1, 0)), ("LEFT", (0, -1)), ("UP", (-1, 0))]:
        new_row = zero_row + direction[0]
        new_col = zero_col + direction[1]

        if 0 <= new_row < size and 0 <= new_col < size:
            new_index = new_row * size + new_col
            new_state = list(current_state)
            new_state[zero_index], new_state[new_index] = new_state[new_index], new_state[zero_index]
            neighbors.append((tuple(new_state), move))
    
    return neighbors

def aStar(start: Tuple[int], goal: Tuple[int], heuristic: Callable[[Tuple[int]], int]) -> Tuple[List[str], int]:
    """
    A* algorithm to find the shortest path from start to goal state
    """
    open_set = heapdict()
    open_set[start] = heuristic(start)

    g = {start: 0}
    came_from = {}
    closed_set = set()

    while open_set:
        current, _ = open_set.popitem()

        if current in closed_set:
            continue

        closed_set.add(current)

        if current == goal:
            return reconstruct_path(came_from, current), len(closed_set)

        for neighbor, move in calculate_neighboring_states(current):
            tentative_g = g[current] + 1

            if neighbor in closed_set and tentative_g >= g.get(neighbor, float("inf")):
                continue

            if tentative_g < g.get(neighbor, float("inf")):
                g[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor)
                open_set[neighbor] = f
                came_from[neighbor] = (current, move)

    return [], len(closed_set)

def reconstruct_path(prev: dict, current: Tuple[int]) -> List[str]:
    """
    Reconstruct the path from start to goal state
    """
    path = []
    while current in prev:
        current, move = prev[current]
        path.append(move)
    
    return path[::-1]

def save_to_file(filename: str, data: List[str]) -> None:
    """
    Save the results to a file
    """
    with open (filename, "a") as file:
        for d in data:
            file.write(" ".join(d) + "\n")

def test(n=10):
    final_state = generate_solved_puzzle(size)
    heuristic = Heuristics(final_state, size)
    result = []

    for i in range(n):
        puzzle = generate_puzzle(size)
        #puzzle = generate_puzzle_x_states_ago(200)
        #puzzle = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 14, 15)
        #puzzle = (12, 1, 10, 2, 7, 11, 4, 14, 5, 9, 15, 8, 13, 3, 6, 0)
        #puzzle = (10, 15, 14, 13, 3, 8, 7, 9, 11, 2, 5, 12, 4, 6, 1, 0)
        print(is_solvable(puzzle))
        print(f"---------Test {i + 1}---------")
        print("Puzzle:")
        for x in range(0, size*size, size):
            print(puzzle[x:x+size])
        #for name, current_heuristic in [("manhattan_linear", heuristic.heuristic_manhattan_linear_conflict), ("manhattan", heuristic.heuristic_manhattan), ("misplaced", heuristic.heuristic_misplaced)]:
        for name, current_heuristic in [("manhattan_linear", heuristic.heuristic_manhattan_linear_conflict), ("manhattan", heuristic.heuristic_manhattan)]:
            start_time = time.time()
            path, visited = aStar(puzzle, final_state, current_heuristic)
            duration = time.time() - start_time
            print(f"\nHeuristic: {name}")
            print(f"Solution: {path}")
            print(f"Number of steps: {len(path)}")
            print(f"Number of visited states: {visited}")
            print(f"Time: {duration:.4f} s")
            result.append([name, str(len(path)), str(visited), str(duration)])
    current_time = datetime.now().strftime("%H:%M:%S")
    name = f"result_{current_time}.txt"
    save_to_file(name, result)

if __name__ == "__main__":
    test(1)