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

    controller = Controller(save_path_search=path + "src/state_recognition/model_saves/segmentation18.pth",
                            save_path_detect=path + "src/state_recognition/model_saves/yolo8n_608_1200.pt",
                            save_path_sgf=path + './sgf.sgf',
                            device='cpu',
                            global_timestamp=global_timestamp)

    window = Window(controller=controller, global_timestamp=global_timestamp, cache_path=path + 'temp/', fps_save=0.1)
    del controller
    window.show()
    sys.exit(app.exec_())

