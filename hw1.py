import copy
from heapq import heappop, heappush
from itertools import chain
import math
import pickle
import random
import sys

import matplotlib.pyplot as plt


class Solver:
    """Solver for n-puzzles.

    Attributes:
        goal_state_1 (list), goal_state_2 (list): Goal states for heuristic calculations only.

    Args:
        start_state (Puzzle): Start state.

    """

    goal_state_1 = None
    goal_state_2 = None

    def __init__(self, start_state):
        self.start_state = start_state
        self.frontier = []  # Visited nodes.
        self.expanded = set()   # Expanded nodes.
        # Order of insertion into frontier, tie-breaks for nodes with same priorities, especially with uniform cost search.
        self.num_visited = 0

    # Solve the puzzle given by the start state using the heuristic provided.
    def solve(self, heuristic):
        root = Node(self.start_state, None, 0, [])
        if not root.state.is_solvable():    # Unsolvable.
            return None
        # Goal states only needed for heuristics.
        if heuristic != Solver.uniform_cost:
            tiles = list(range(1, root.state.width**2))
            Solver.goal_state_1 = [0] + tiles
            Solver.goal_state_2 = tiles + [0]
        self.num_visited += 1
        heappush(self.frontier, (root.depth +
                                 heuristic(root), self.num_visited, root))
        while self.frontier:
            cur_node = heappop(self.frontier)[-1]
            if cur_node.state.is_goal_state():  # Solved!
                solution = self.solved(heuristic, cur_node)
                return solution
            # Tuples are hashable.
            cur_position = tuple(cur_node.state.position_flat)
            # Avoid expanding nodes already expanded.
            if cur_position not in self.expanded:
                self.expanded.add(cur_position)
                for direction in cur_node.state.possible_moves():
                    new_node = copy.deepcopy(cur_node)
                    new_node.state.move_blank(direction)    # Expansion.
                    # Only visit nodes not expanded yet.
                    if tuple(new_node.state.position_flat) not in self.expanded:
                        self.num_visited += 1
                        new_node.parent = cur_node  # Add into the search tree.
                        new_node.depth += 1
                        new_node.path.append(direction)
                        heappush(self.frontier, (new_node.depth +
                                                 heuristic(new_node), self.num_visited, new_node))

    # Return a Solution object populated with information about the solution.
    def solved(self, heuristic, goal_node):
        return Solution(self.start_state, heuristic, len(self.expanded), goal_node.state, goal_node.path)

    # Return heuristic for uniform cost search.
    @staticmethod
    def uniform_cost(node):
        return 0

    # Return misplaced tiles heuristic.
    @staticmethod
    def tiles(node):
        # Take the minimum of the forward cost to either of the goal states to make sure heuristic is admissible.
        return min(Solver.misplaced_tiles(node, Solver.goal_state_1), Solver.misplaced_tiles(node, Solver.goal_state_2))

    # Return misplaced tiles heuristic with regard to a specific goal state.
    @staticmethod
    def misplaced_tiles(node, goal_state):
        return compare_arrays(node.state.position_flat, goal_state)

    # Return Manhattan distance heuristic.
    @staticmethod
    def manhattan(node):
        # Take the minimum of the forward cost to either of the goal states to make sure heuristic is admissible.
        return min(Solver.manhattan_heuristic(node, Solver.goal_state_1), Solver.manhattan_heuristic(node, Solver.goal_state_2))

    # Return Manhattan distance heuristic with regard to a specifc goal state.
    @staticmethod
    def manhattan_heuristic(node, goal_state):
        width = node.state.width
        distance = 0
        for i in range(width):
            for j in range(width):
                cur_tile = node.state.position[i][j]
                if cur_tile != 0:
                    # Find the position of the current tile in the goal state.
                    index = goal_state.index(cur_tile)
                    # Calculate coordinates of it in 2D representation.
                    row = index // width
                    col = index % width
                    distance += manhattan_distance((i, j), (row, col))
        return distance


class Solution():
    """Represent a solution of a n-puzzle.

    Args:
        start_state (Puzzle): Start state.
        heuristic (func): Heuristic function.
        num_expanded (int): Number of expanded nodes.
        sol_state (Puzzle): End state.
        sol_path (list): List of movements of blank tiles to reach end state.

    """

    def __init__(self, start_state, heuristic, num_expanded, sol_state, sol_path):
        self.start_state = start_state
        self.heuristic = heuristic
        self.num_expanded = num_expanded
        self.sol_state = sol_state
        self.sol_path = sol_path

    def __str__(self):
        # Pretty display of the solution.
        solution_str = []
        solution_str.append('\nHeuristic: {}'.format(self.heuristic.__name__))
        solution_str.append(
            'Number of expanded nodes: {}'.format(self.num_expanded))
        solution_str.append('\nStart state:')
        solution_str.append('\n{}\n'.format(self.start_state))
        solution_str.append('Width: n = {}'.format(self.start_state.width))
        solution_str.append('Solution state:')
        solution_str.append('\n{}\n'.format(self.sol_state))
        solution_str.append(
            'Solution path depth: {}'.format(len(self.sol_path)))
        solution_str.append('Solution path:')
        solution_str.append(','.join(self.sol_path))
        return '\n'.join(solution_str)


class Node:
    """Represent a node in the search tree.

    Args:
        state (Puzzle): State of the position.
        parent (Node): Parent node.
        depth (int): Depth of node in search tree.
        path (list): List of movements of blank tiles to reach this state.

    """

    def __init__(self, state, parent, depth, path):
        self.state = state
        self.parent = parent
        self.depth = depth
        self.path = path


class Puzzle:
    """Represent a n-puzzle.

    Args:
        position (list): Row-major representation of the puzzle.

    """

    def __init__(self, position):
        self.position_flat = position
        self.width = int(math.sqrt(len(position)))
        # 2D representation of the puzzle.
        self.position = reshape(position, self.width)
        # Indices of blank tile in 2D representation.
        self.blank_ix = self.find(0)

    def __str__(self):
        # Return pretty 2D representation of the puzzle
        return '\n'.join([' '.join(list(map(str, row))) for row in self.position]).replace('0', ' ')

    # Find coordinates of blank tile in 2D representation of the puzzle.
    def find(self, tile):
        for i in range(self.width):  # For each row.
            try:
                j = self.position[i].index(tile)    # Find column index in row.
                return i, j
            except ValueError:  # Not found in current row.
                continue

    # Check if puzzle is solvable. For odd-numbered puzzle, the position is unsolvable if the number of inversions is odd.
    def is_solvable(self):
        if self.width % 2 != 0 and self.count_inversions() % 2 != 0:
            return False
        return True

    # Count the number of inversions in the puzzle. An inversion is when a bigger-numbered tile appears before a smaller-numbered tile.
    def count_inversions(self):
        # Use flat representation for convenience.
        position = self.position_flat
        count = 0
        for i in range(self.width**2):
            if position[i] != 0:    # Blank tile doesn't count in inversions.
                # Traverse through the tiles which appear after this tile.
                for j in range(i+1, self.width**2):
                    if position[j] != 0 and position[i] > position[j]:
                        count += 1
        return count

    # Return list of possible moves of blank tiles for the current position.
    def possible_moves(self):
        moves = []
        row, col = self.blank_ix
        if row > 0:
            moves.append('U')
        if row < self.width - 1:
            moves.append('D')
        if col > 0:
            moves.append('L')
        if col < self.width - 1:
            moves.append('R')
        return moves

    # Move the blank tile in the specified position.
    def move_blank(self, direction):
        row, col = self.blank_ix
        if direction == 'U':
            swap_elements(self.position, self.blank_ix, (row-1, col))
            row -= 1
        elif direction == 'D':
            swap_elements(self.position, self.blank_ix, (row+1, col))
            row += 1
        elif direction == 'L':
            swap_elements(self.position, self.blank_ix, (row, col-1))
            col -= 1
        else:
            swap_elements(self.position, self.blank_ix, (row, col+1))
            col += 1
        self.blank_ix = (row, col)  # Update the blank tile's position.
        # Update the flat representation of the puzzle.
        self.position_flat = flatten(self.position)

    # Transform the current puzzle using the specified move sequence. Mostly for debugging and verification.
    def move_sequence(self, sequence, verbose=False):
        if verbose:
            print(self)
        for move in sequence:
            self.move_blank(move)
            if verbose:
                print('\n' + move + '\n')
                print(self)
        if not verbose:
            print(self)
        print('\n' + str(self.is_goal_state()))

    # Check if the position is a goal state.
    def is_goal_state(self):
        state = self.position_flat  # Use flat representation for convenience.
        # Goal states always have blank tile at the first or last position.
        if state[0] != 0 and state[-1] != 0:
            return False
        for i in range(self.width**2 - 1):
            if state[i+1] != 0 and state[i] > state[i+1]:
                return False
        return True

    # Generate a random Puzzle of the specified width.
    @staticmethod
    def generate(width):
        tiles = list(range(width**2))   # Possible tiles.
        random.shuffle(tiles)
        return Puzzle(tiles)


# Return 2D representation of the puzzle given the flat array and width.
def reshape(array, width):
    return [array[i:i+width] for i in range(0, width**2, width)]


# Flatten a 2D representation into a 1D array.
def flatten(array_2d):
    return list(chain.from_iterable(array_2d))


# Swap 2 elements in a 2D array, given the indices.
def swap_elements(array_2d, ix1, ix2):
    row1, col1 = ix1
    row2, col2 = ix2
    array_2d[row1][col1], array_2d[row2][col2] = array_2d[row2][col2], array_2d[row1][col1]


# Count the differences in two 1D arrays.
def compare_arrays(array_1, array_2):
    # Does not work with arrays of different lengths.
    if len(array_1) != len(array_2):
        return -1
    return sum(1 if array_1[i] != array_2[i] else 0 for i in range(len(array_1)))


# Calculate the Manhattan distance between 2 points given the 2D coordinates.
def manhattan_distance(point_1, point_2):
    x_1, y_1 = point_1
    x_2, y_2 = point_2
    return math.fabs(x_1 - x_2) + math.fabs(y_1 - y_2)


# Solve num_instances of randomly generated puzzles of specified width using specified heuristics.
# Return the list of solution depths and a dictionary of the number of expanded nodes for each heuristic.
def solve_n_times(width, heuristics, num_instances):
    sol_depths = []
    num_expanded_dict = dict()
    # Initializing the dictionary for each heuristic.
    for heuristic in heuristics:
        num_expanded_dict[heuristic.__name__] = []
    while num_instances > 0:
        print('Number of instances left: {}'.format(num_instances))
        start_state = Puzzle.generate(3)
        if start_state.is_solvable():   # Ignore unsolvable puzzles.
            solution = None
            for heuristic in heuristics:
                print('Heuristic: {}'.format(heuristic.__name__))
                solver = Solver(start_state)
                solution = solver.solve(heuristic)
                num_expanded_dict[heuristic.__name__].append(
                    solution.num_expanded)
            # Same depth for all heuristics due to optimality.
            sol_depths.append(len(solution.sol_path))
            num_instances -= 1
    return sol_depths, num_expanded_dict


# Plot the results of solve_n_times from data in the specified location and save the figure.
def plot_num_expanded_dict(sol_depths, num_expanded_dict):
    plt.figure()
    plt.title('Number of Nodes Expanded Vs. Goal State Depth')
    plt.xlabel('Goal State Depth')
    plt.ylabel('Number of Nodes Expanded')
    colors = ['b', 'g', 'r']
    for heuristic, color in zip(num_expanded_dict.keys(), colors[:len(num_expanded_dict)]):
        plt.plot(sol_depths, num_expanded_dict[heuristic], '{}.'.format(
            color), label=heuristic)
    plt.legend(loc='best')
    plt.savefig('algorithm_efficiency.png', bbox_inches='tight')
    plt.show()


def main():
    if len(sys.argv) < 2:
        err_msg = 'Missing argument! Syntax: python hw1.py input_file [heuristic] or python hw1.py -r [num_instances]'
        sys.exit(err_msg)
    # Mass solving of random puzzles.
    if sys.argv[1] == '-r' and len(sys.argv) == 3:
        num_instances = int(sys.argv[2])
        print('Solving randomly generated {} 8-puzzles and plotting results...'.format(num_instances))
        heuristics = [Solver.uniform_cost, Solver.tiles, Solver.manhattan]
        sol_depths, num_expanded_dict = solve_n_times(
            3, heuristics, num_instances)
        # Pickle the results for use later.
        with open('sol_depths.pkl', 'wb') as f:
            pickle.dump(sol_depths, f)
        with open('num_expanded_dict.pkl', 'wb') as f:
            pickle.dump(num_expanded_dict, f)
        print('Finished solving puzzles!')
        print(sol_depths)
        print(num_expanded_dict)
        plot_num_expanded_dict(sol_depths, num_expanded_dict)
    else:
        if sys.argv[1] == '-r':  # Solve just 1 random puzzle.
            print('Solving randomly generated 8-puzzle...')
            start_state = Puzzle.generate(3)
            heuristics = [Solver.uniform_cost, Solver.tiles, Solver.manhattan]
        else:
            # Default heuristic is Manhattan distance.
            if len(sys.argv) < 3 or sys.argv[2] == 'manhattan':
                heuristics = [Solver.manhattan]
            elif sys.argv[2] == 'ucs':
                heuristics = [Solver.uniform_cost]
            elif sys.argv[2] == 'tiles':
                heuristics = [Solver.tiles]
            elif sys.argv[2] == 'all':
                heuristics = [Solver.uniform_cost,
                              Solver.tiles, Solver.manhattan]
            else:
                err_msg = "Invalid heuristic! Please select either 'ucs', 'tiles, 'manhattan', or 'all'."
                sys.exit(err_msg)
            input_file = sys.argv[1]
            print("Opening input file '{}'...".format(input_file))
            with open(input_file, 'r') as f:
                raw_puzzle = f.readline()
            start_state = Puzzle(list(map(int, raw_puzzle[:-1].split(','))))
        for heuristic in heuristics:
            solver = Solver(start_state)
            solution = solver.solve(heuristic)
            if solution is None:
                print('Unsolvable')
                break
            else:
                print(solution)
            # # Verify correctness of solution step-by-step.
            # start_state.move_sequence(solution.sol_path, True)


if __name__ == '__main__':
    main()
