from PyQt5.QtWidgets import QApplication
import sys
from multiprocessing import Value

app = QApplication(sys.argv)
from src.gui import Controller
from src.gui import Window

global_timestamp = Value('i', 0)

controller = Controller("src/state_recognition/model_saves/segmentation18.pth",
                        "src/state_recognition/model_saves/yolo8n_320.pt", device='cpu',
                        global_timestamp=global_timestamp)

window = Window(controller=controller, global_timestamp=global_timestamp)
del controller
window.show()
sys.exit(app.exec_())
