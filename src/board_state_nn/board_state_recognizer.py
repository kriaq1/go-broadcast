import torch
from .src import get


class BoardStateRecognizer:
    def __init__(self, save_path='configs/model_saves/board_state.pth', device=torch.device('cpu')):
        self.device = device
        self.model = get.model(load_path=save_path, device=self.device)

    def get_predict(self, input, out_threshold=0.5):
        return get.predict_input(input, self.model, self.device, out_threshold=out_threshold)

    def get_predicts(self, inputs, out_threshold=0.5):
        return get.predict_inputs(inputs, self.model, self.device, out_threshold=out_threshold)


if __name__ == "__main__":
    device = torch.device('cpu')
    predictor = BoardStateRecognizer('model.pth', device)

    from os import listdir

    for file in listdir('test/input/'):
        input = get.input('test/input/' + str(file))

        from src.utils import save_prediction

        save_prediction(input, predictor.get_predict(input), 'test/result/')
