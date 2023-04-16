def get_board_state(result, min_distance, max_distance):
    boxes_n = result.boxes.xywhn
    conf = result.boxes.conf
    cls = result.boxes.cls

    coordinates, classes, probabilities = preprocess_boxes(boxes_n, conf, cls)
    graph = build_graph(coordinates, classes, min_distance, max_distance)
    subgraph = find_subgraph(graph)


def preprocess_boxes(boxes_n, conf, cls):
    coordinates = []
    classes = []
    probabilities = []
    for box, prob, c in zip(boxes_n, conf, cls):
        coordinates.append(box[:2])
        classes.append(c)
    return coordinates, classes, probabilities


def build_graph(coordinates, classes, probabilities, min_distance, max_distance):
    pass


def find_subgraph(graph):
    pass
