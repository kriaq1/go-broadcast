import numpy as np
from collections import deque


def get_board_state(result, min_distance, max_distance) -> tuple[np.ndarray, np.ndarray]:
    try:
        result = result.cpu()
        boxes_n = result.boxes.xywhn
        conf = result.boxes.conf
        cls = result.boxes.cls

        coordinates, classes, probabilities = preprocess_boxes(boxes_n, cls, conf, min_distance)
        graph = build_graph(coordinates, classes, probabilities, min_distance, max_distance)
        component = get_largest_component(graph)
        relative_coordinates = get_relative_coordinates(graph, component)
        board, probabilities = find_subgraph(relative_coordinates)
        return board - 1, probabilities
    except Exception:
        return np.zeros((19, 19), dtype=int), np.zeros((19, 19))


def preprocess_boxes(boxes_n, cls, conf, min_distance):
    coordinates = []
    classes = []
    probabilities = []
    boxes_n = boxes_n.numpy().astype(float)
    cls = cls.numpy().astype(int)
    conf = conf.numpy().astype(float)
    min_box_area = min_distance * min_distance * 0.75
    for box, c, prob in zip(boxes_n, cls, conf):
        width = float(box[2])
        height = float(box[3])
        if not (0.75 <= width / height <= 1.5) or width * height < min_box_area:
            continue
        coordinates.append((box[0], box[1]))
        classes.append(c)
        probabilities.append(prob)
    return np.array(coordinates), np.array(classes), np.array(probabilities)


def check_edge(distance):
    distance_x = float(distance[0])
    distance_y = float(distance[1])

    if abs(distance_x) < abs(distance_y) and distance_y > 0:
        return 0
    elif abs(distance_x) > abs(distance_y) and distance_x > 0:
        return 1
    elif abs(distance_x) < abs(distance_y) and distance_y < 0:
        return 2
    else:
        return 3


def build_graph(coordinates, classes, probabilities, min_dist, max_dist) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = coordinates.shape[0]
    distances = np.reshape(coordinates, (size, 1, 2)) - np.reshape(coordinates, (1, size, 2))
    abs_distances = np.sum(np.abs(distances), axis=2)
    argsort_distances = np.argsort(abs_distances, axis=1)
    edges = np.zeros((size, 4), dtype=int) - 1
    groups = []
    for i in range(size):
        groups.append([])
        for j in argsort_distances[i]:
            if i == j:
                continue
            distance = float(abs_distances[i][j])
            if distance > max_dist:
                break
            if distance < min_dist:
                groups[i].append(j)
                continue
            direction = check_edge(distances[i][j])
            if edges[i][direction] == -1:
                edges[i][direction] = j
    for i in range(size):
        for j in groups[i]:
            old_index = j
            new_index = i
            if probabilities[new_index] < probabilities[old_index]:
                old_index, new_index = new_index, old_index
            edges[edges == old_index] = new_index
    return classes, probabilities, edges


def get_largest_component(graph):
    classes = graph[0]
    edges = graph[2]
    size = len(classes)
    used = np.zeros(size, dtype=bool)

    def dfs(vertex, count: list):
        used[vertex] = True
        count[0] += 1
        for next_vertex in edges[vertex]:
            if next_vertex != -1 and not used[next_vertex]:
                dfs(next_vertex, count)

    best, result = 0, 0
    for vertex in np.random.permutation(size):
        if not used[vertex]:
            count = [0]
            dfs(vertex, count)
            if best < count[0]:
                best, result = count[0], vertex
    return result


def get_relative_coordinates(graph, component, board_size=19) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    classes = graph[0]
    probabilities = graph[1]
    edges = graph[2]
    size = len(classes)

    relative_coordinates = [None] * size
    queue = deque()
    queue.append(component)
    relative_coordinates[component] = 0, 0
    while queue:
        vertex = queue.popleft()
        x, y = relative_coordinates[vertex]
        diff = [(0, -1), (-1, 0), (0, 1), (1, 0)]
        for edge in range(4):
            next_vertex = edges[vertex][edge]
            if next_vertex != -1 and relative_coordinates[next_vertex] is None:
                relative_coordinates[next_vertex] = x + diff[edge][0], y + diff[edge][1]
                queue.append(next_vertex)
    relative_coordinates = np.array(relative_coordinates, dtype=object)
    not_none_coordinates = np.array(relative_coordinates[relative_coordinates != None].tolist())
    min_element = np.min(not_none_coordinates)
    max_element = max(board_size, np.max(not_none_coordinates) - min_element + 1)
    result_mask = np.zeros((max_element, max_element), dtype=int)
    result_classes = np.ones((max_element, max_element), dtype=int)
    result_probabilities = np.zeros((max_element, max_element), dtype=float)
    for i in range(size):
        coordinate = relative_coordinates[i]
        if coordinate is not None:
            x, y = coordinate
            x -= min_element
            y -= min_element
            result_mask[y][x] = 1
            result_classes[y][x] = classes[i]
            result_probabilities[y][x] = probabilities[i]
    return result_mask, result_classes, result_probabilities


def find_subgraph(relative_coordinates, board_size=19) -> tuple[np.ndarray, np.ndarray]:
    mask = relative_coordinates[0]
    classes = relative_coordinates[1]
    probabilities = relative_coordinates[2]
    size = mask.shape[0]

    best = 0
    x, y = 0, 0
    for i in range(size - board_size + 1):
        for j in range(size - board_size + 1):
            cut_mask_sum = np.sum(mask[i:i + board_size, j:j + board_size])
            if best < cut_mask_sum:
                best = cut_mask_sum
                y, x = i, j
    return classes[y:y + board_size, x:x + board_size], probabilities[y:y + board_size, x:x + board_size]
