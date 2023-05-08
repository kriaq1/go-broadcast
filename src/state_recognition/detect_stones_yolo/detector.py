from ultralytics import YOLO
import torch


class StoneDetector:
    def __init__(self, device='cpu', save_path='configs/model_saves/yolo8m.pth'):
        self.device = str(device) if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(save_path)
        self.model.to(self.device)

    def get_predict(self, source, conf=0.25, iou=0.5, max_det=1000, mode='predict'):
        device = self.device if self.device != 'cuda' else None
        results = self.model.predict(source=source,
                                     save=False,
                                     save_txt=False,
                                     conf=conf,
                                     iou=iou,
                                     max_det=max_det,
                                     line_width=1,
                                     show=False,
                                     show_labels=False,
                                     show_conf=False,
                                     device=device,
                                     mode=mode)
        return results

    def get_track(self, source, conf=0.25, iou=0.5, max_det=1000):
        return self.get_predict(source, conf, iou, max_det, mode='track')
