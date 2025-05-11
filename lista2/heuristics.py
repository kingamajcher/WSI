from typing import Tuple

class Heuristics:
    def __init__(self, final_state: Tuple[int], puzzle_size: int):
        self.final_state = final_state
        self.puzzle_size = puzzle_size
        self.goal_position = {value: (i // puzzle_size, i % puzzle_size) for i, value in enumerate(final_state)}

    def heuristic_misplaced(self, state: Tuple[int]) -> int:
        """
        Heuristic counting how many numbers are misplaced
        """
        return sum(1 for i in range(self.puzzle_size ** 2) if state[i] != 0 and state[i] != self.final_state[i])
    
    def heuristic_manhattan(self, state: Tuple[int]) -> int:
        """
        Manhattan heuristic (|x1 - x2| + |y1 - y2|)
        """
        distance = 0
        for index, value in enumerate(state):
            if value == 0:
                continue
            curr_row, curr_col = divmod(index, self.puzzle_size)
            goal_row, goal_col = self.goal_position[value]
            distance += abs(curr_row - goal_row) + abs(curr_col - goal_col)
        return distance
    
    def heuristic_manhattan_linear_conflict(self, state: Tuple[int]) -> int:
        """
        Manhattan heuristic with linear conflict
        """
        size = self.puzzle_size
        manhattan = self.heuristic_manhattan(state)
        linear_conflict = 0
    
        for row in range(size):
            max_goal_col = -1
            for col in range(size):
                index = row * size + col
                value = state[index]
                if value == 0:
                    continue
                goal_row, goal_col = self.goal_position[value]
                if goal_row == row:
                    if goal_col > max_goal_col:
                        max_goal_col = goal_col
                    else:
                        linear_conflict += 1
    
        for col in range(size):
            max_goal_row = -1
            for row in range(size):
                index = row * size + col
                value = state[index]
                if value == 0:
                    continue
                goal_row, goal_col = self.goal_position[value]
                if goal_col == col:
                    if goal_row > max_goal_row:
                        max_goal_row = goal_row
                    else:
                        linear_conflict += 1
    
        return manhattan + 2 * linear_conflict

    

