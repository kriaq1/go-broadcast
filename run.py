from PyQt5.QtWidgets import QApplication
import sys
from multiprocessing import Value

app = QApplication(sys.argv)
from src.gui import Controller
from src.gui import Window

import os

if __name__ == '__main__':
    path = os.path.abspath(__file__)
    path = path.replace(os.sep, '/')
    path = os.path.split(path)[0] + '/'

    global_timestamp = Value('i', 0)
    recognition_parameters = dict(search_period=4,
                                  seg_threshold=0.5,
                                  quality_coefficient=0.975,
                                  conf=0.4,
                                  iou=0.5,
                                  min_distance=20 / 608,
                                  max_distance=50 / 608)

    logging_parameters = dict(delay=1,
                              appearance_count=3,
                              valid_time=3 * 1000,
                              valid_zeros_count=20,
                              valid_thresh=0.2)
    recognition_parameters = "configs/recognition_config.yml"
    logging_parameters = "configs/logging_config.yml"
    text_info = ""
    if os.path.isfile("info.md"):
        with open("info.md", encoding="utf8") as fd:
            text_info = fd.read(-1)
    controller = Controller(save_path_search=path + "src/state_recognition/model_saves/segmentation18.pth",
                            save_path_detect=path + "src/state_recognition/model_saves/yolo8n_608_1200.pt",
                            save_path_sgf=path + './sgf.sgf',
                            device='cpu',
                            global_timestamp=global_timestamp)

    window = Window(controller=controller, global_timestamp=global_timestamp, cache_path=path + 'temp/', fps_save=0.1,
                    recognition_parameters=recognition_parameters, logging_parameters=logging_parameters, info_text=text_info)
    del controller
    window.show()
    sys.exit(app.exec_())
