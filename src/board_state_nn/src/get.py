import torch
import numpy as np

from .model import Model

from . import preprocess


# return initialized model
def model(load_path, device):
    initialize_weights = False if load_path else True
    model = Model(out_classes=3,
                  initialize_weights=initialize_weights)

    if load_path:
        state_dict = torch.load(load_path, map_location=device)
        model.load_state_dict(state_dict)

    model.to(device=device)
    return model


# return predicted list of targets
def predict_inputs(inputs, model, device, out_threshold=0.5):
    model.eval()
    for i in range(len(inputs)):
        inputs[i] = preprocess_input(inputs[i])
    input = torch.from_numpy(np.array(inputs))
    input = input.to(device, dtype=torch.float32)
    with torch.no_grad():
        output = model(input).cpu()
        predict = output.argmax(dim=1)
    return predict.long().squeeze(1).numpy().astype(int) - 1


# return predicted target
def predict_input(input, model, device, out_threshold=0.5):
    return predict_inputs([input], model, device, out_threshold)[0]


# open input by path in a certain format
def input(filename):
    return preprocess.open_input(filename)


# open target by path in a certain format
def target(filename):
    return preprocess.open_target(filename)


# preprocess or assert input for neural network. Called after method 'input' or in predict
def preprocess_input(input):
    return preprocess.preprocess_input(input)


# preprocess target for neural network. Called after method 'target'
def preprocess_target(target):
    return preprocess.preprocess_target(target)
