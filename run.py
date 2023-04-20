from src.app import App
from src.asciiapi import ASCIIDump
import torch
import pathlib


working_path = str(pathlib.Path(__file__).parent.resolve())

app = App(
    [ASCIIDump("/tmp/gotest")],
    working_path + '/configs/model_saves/segmentation.pth',
    working_path + '/configs/model_saves/yolo8s.pt',
    torch.device('cpu')
)

app.start()
