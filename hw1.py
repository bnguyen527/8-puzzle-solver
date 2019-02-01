from itertools import chain
import math


class Puzzle:
    def __init__(self, puzzle):
        self.state_flat = puzzle
        self.width = int(math.sqrt(len(puzzle)))
        self.state = reshape(puzzle, self.width)

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


def reshape(array, width):
    return [array[i:i+width] for i in range(0, width**2, width)]


def flatten(array_2d):
    return list(chain.from_iterable(array_2d))


def main():
    with open('puzzle1.csv', 'r') as f:
        puzzle_input = f.readline()
    puzzle_input = list(map(int, puzzle_input[:-1].split(',')))
    puzzle = Puzzle(puzzle_input)
    print(puzzle.is_solvable())


if __name__ == '__main__':
    main()
