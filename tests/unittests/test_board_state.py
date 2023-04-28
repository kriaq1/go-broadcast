import pytest
from src.state_recognition.board_state import *


@pytest.mark.test_board_state
def test_board_state():
    coordinates = np.array(
        [[2, 2], [2, 2], [1.9, 2], [2.9, 3], [3.1, 3], [0, 4], [0.1, 4], [3, 2], [2, 3], [3, 4], [0, 0], [1, 1], [3, 0],
         [3, 1], [4, 1], [4, 2], [5, 3], [2, 4], [4, 4], [5, 4], [2, 5]])

    classes = np.array([1, 1, 1, 2, 1, 1, 2, 0, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    probabilities = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    graph = build_graph(coordinates, classes, probabilities, 0.5, 1.5)

    assert len(graph[0]) == len(graph[1]) == len(graph[2])
    assert graph[2].shape == (len(graph[2]), 4)

    component = get_largest_component(graph)

    assert 0 <= component < len(graph[0])

    relative_coordinates = get_relative_coordinates(graph, component, board_size=4)

    board, probabilities = find_subgraph(relative_coordinates, board_size=4)

    true_board = np.array([[1, 1, 1, 1], [1, 0, 1, 1], [2, 1, 1, 1], [1, 0, 1, 1]])
    true_probabilities = np.array([[0, 1, 1, 0], [0.5, 1, 1, 0], [1, 0.5, 0, 1], [1, 1, 1, 1]])
    assert (board == true_board).any()
    assert (probabilities == true_probabilities).any()
