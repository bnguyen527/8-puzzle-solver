import copy
from heapq import heappop, heappush
from itertools import chain
import math


class Solver:
    def __init__(self, start_state):
        self.start_state = start_state
        self.frontier = []
        self.expanded = set()
        self.num_visited = 0
    
    def solve(self):
        root = Node(self.start_state, None, 0, [])
        if not root.state.is_solvable():
            return None
        self.num_visited += 1
        heappush(self.frontier, (root.depth, self.num_visited, root))
        while self.frontier:
            cur_node = heappop(self.frontier)[-1]
            if cur_node.state.is_goal_state():
                return cur_node
            cur_position = tuple(cur_node.state.position_flat)
            if cur_position not in self.expanded:
                self.expanded.add(cur_position)
                for direction in cur_node.state.possible_moves():
                    new_node = copy.deepcopy(cur_node)
                    new_node.state.move_blank(direction)
                    if tuple(new_node.state.position_flat) not in self.expanded:
                        self.num_visited += 1
                        new_node.parent = cur_node
                        new_node.depth += 1
                        new_node.path.append(direction)
                        heappush(self.frontier, (new_node.depth, self.num_visited, new_node))


class Node:
    def __init__(self, state, parent, depth, path):
        self.state = self.Puzzle(state)
        self.parent = parent
        self.depth = depth
        self.path = path
    
    class Puzzle:
        def __init__(self, position):
            self.position_flat = position
            self.width = int(math.sqrt(len(position)))
            self.position = reshape(position, self.width)
            self.blank_ix = self.find(0)

        def __str__(self):
            return ('\n'.join([' '.join(list(map(str, row))) for row in self.position])).replace('0', ' ')

        def find(self, tile):
            for i in range(self.width):
                try:
                    j = self.position[i].index(tile)
                    return i, j
                except ValueError:
                    continue

        def is_solvable(self):
            if self.width % 2 != 0 and self.count_inversions() % 2 != 0:
                return False
            return True

        def count_inversions(self):
            position = self.position_flat
            count = 0
            for i in range(self.width**2):
                if position[i] != 0:
                    for j in range(i+1, self.width**2):
                        if position[j] != 0 and position[i] > position[j]:
                            count += 1
            return count

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
            self.blank_ix = (row, col)
            self.position_flat = flatten(self.position)
        
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

        def is_goal_state(self):
            state = self.position_flat
            if state[0] != 0 and state[-1] != 0:
                return False
            for i in range(self.width**2 - 1):
                if state[i+1] != 0 and state[i] > state[i+1]:
                    return False
            return True


def reshape(array, width):
    return [array[i:i+width] for i in range(0, width**2, width)]


def flatten(array_2d):
    return list(chain.from_iterable(array_2d))


def swap_elements(array_2d, ix1, ix2):
    row1, col1 = ix1
    row2, col2 = ix2
    array_2d[row1][col1], array_2d[row2][col2] = array_2d[row2][col2], array_2d[row1][col1]


def main():
    with open('test.csv', 'r') as f:
        raw_puzzle = f.readline()
    start_state = list(map(int, raw_puzzle[:-1].split(',')))
    solver = Solver(start_state)
    solution = solver.solve()
    if solution is None:
        print('Unsolvable')
    else:
        print(solution.path)
    # move_sequence = ['U', 'L', 'U', 'R', 'D', 'L', 'D', 'R', 'U', 'R', 'D', 'L', 'L', 'U', 'R']
    puzzle = Node.Puzzle(start_state)
    puzzle.move_sequence(solution.path, True)


if __name__ == '__main__':
    main()
