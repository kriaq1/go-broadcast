from PyQt5.QtWidgets import QApplication
import sys
from multiprocessing import Value

app = QApplication(sys.argv)
from src.gui import Controller
from src.gui import Window

global_timestamp = Value('i', 0)

controller = Controller(save_path_search="src/state_recognition/model_saves/segmentation18.pth",
                        save_path_detect="src/state_recognition/model_saves/yolo8n_608_1200.pt",
                        save_path_sgf='./sgf.sgf',
                        device='cpu',
                        global_timestamp=global_timestamp)

window = Window(controller=controller, global_timestamp=global_timestamp)
del controller
window.show()
sys.exit(app.exec_())
