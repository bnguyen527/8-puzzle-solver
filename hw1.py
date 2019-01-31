import math


class Puzzle:
    def __init__(self, puzzle):
        self.width = int(math.sqrt(len(puzzle)))
        self.state = reshape(puzzle, self.width)

    def find(self, tile):
        for i in range(self.width):
            try:
                j = self.state[i].index(tile)
                return i, j
            except ValueError:
                continue


def reshape(array, width):
    return [array[i:i+width] for i in range(0, width**2, width)]


def main():
    with open('puzzle1.csv', 'r') as f:
        puzzle_input = f.readline()
    puzzle_input = list(map(int, puzzle_input[:-1].split(',')))
    puzzle = Puzzle(puzzle_input)
    print(puzzle.find(0))


if __name__ == '__main__':
    main()
