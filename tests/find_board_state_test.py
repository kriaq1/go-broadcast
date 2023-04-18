from src.state_recognition.board_state import *

if __name__ == '__main__':
    coordinates = np.array(
        [[2, 2], [2, 2], [1.9, 2], [2.9, 3], [3.1, 3], [0, 4], [0.1, 4], [3, 2], [2, 3], [3, 4], [0, 0], [1, 1], [3, 0],
         [3, 1], [4, 1], [4, 2], [5, 3], [2, 4], [4, 4], [5, 4], [2, 5]])

    classes = np.array([1, 1, 1, 2, 1, 1, 2, 0, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    probabilities = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    graph = build_graph(coordinates, classes, probabilities, 0.5, 1.5)

    print('Classes:')
    print(graph[0])
    print('Probabilities:')
    print(graph[1])
    print('Edges:')
    print(graph[2])

    component = get_largest_component(graph)

    print('Component:')
    print(component)

    relative_coordinates = get_relative_coordinates(graph, component)
    print('Relative mask:')
    print(relative_coordinates[0])
    print('Relative classes:')
    print(relative_coordinates[1])
    print('Relative probabilities:')
    print(relative_coordinates[2])

    board, probabilities = find_subgraph(relative_coordinates)
    print('Board:')
    print(board)
    print('Probabilities:')
    print(probabilities)
