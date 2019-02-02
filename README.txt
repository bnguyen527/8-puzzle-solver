Python script to solve n-puzzles using A* search.

Heuristics supported:
    'ucs': Uniform-cost search
    'tiles': Number of misplaced tiles
    'manhattan': Manhattan distance
    'all': Use all heuristics, each following the other

Default heuristic is Manhattan distance.

To solve a puzzle stored in a text file:
    python hw1.py input_file [heuristic]

To solve a randomly generated puzzle using all heuristics:
    python hw1.py -r

To solve num_instances randomly generated 8-puzzles using all heuristics and
plot the results:
    python hw1.py -r num_instances

    Additionally, data from the num_instances runs and the algorithm efficiency
    graph are stored in 'sol_depths.pkl', 'num_expanded_dict.pkl', and
    'algorithm_efficiency.png', respectively, in the same directory.

To test the limits of a certain heuristic and get solving time:
    python hw1.py -t heuristic width
