from itertools import chain
import math


class Puzzle:
    def __init__(self, puzzle):
        self.state_flat = puzzle
        self.width = int(math.sqrt(len(puzzle)))
        self.state = reshape(puzzle, self.width)
        self.blank_ix = self.find(0)
    
    def __str__(self):
        return ('\n'.join([' '.join(list(map(str, row))) for row in self.state])).replace('0', ' ')

    def find(self, tile):
        for i in range(self.width):
            try:
                j = self.state[i].index(tile)
                return i, j
            except ValueError:
                continue
    
    def is_solvable(self):
        if self.width % 2 != 0 and self.count_inversions() % 2 != 0:
            return False
        return True
            
    def count_inversions(self):
        state = self.state_flat
        count = 0
        for i in range(self.width**2):
            if state[i] != 0:
                for j in range(i+1, self.width**2):
                    if state[j] != 0 and state[i] > state[j]:
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
            swap_elements(self.state, self.blank_ix, (row-1, col))
            row -= 1
        elif direction == 'D':
            swap_elements(self.state, self.blank_ix, (row+1, col))
            row += 1
        elif direction == 'L':
            swap_elements(self.state, self.blank_ix, (row, col-1))
            col -= 1
        else:
            swap_elements(self.state, self.blank_ix, (row, col+1))
            col += 1
        self.blank_ix = (row, col)
        self.state_flat = flatten(self.state)



def reshape(array, width):
    return [array[i:i+width] for i in range(0, width**2, width)]


def flatten(array_2d):
    return list(chain.from_iterable(array_2d))


def swap_elements(array_2d, ix1, ix2):
    row1, col1 = ix1
    row2, col2 = ix2
    array_2d[row1][col1], array_2d[row2][col2] = array_2d[row2][col2], array_2d[row1][col1]


def main():
    with open('puzzle1.csv', 'r') as f:
        puzzle_input = f.readline()
    puzzle_input = list(map(int, puzzle_input[:-1].split(',')))
    puzzle = Puzzle(puzzle_input)
    print(puzzle)
    print(puzzle.possible_moves())
    puzzle.move_blank('U')
    print(puzzle)


if __name__ == '__main__':
    main()
