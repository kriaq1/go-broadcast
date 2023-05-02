from PyQt5.QtWidgets import QApplication
import sys

app = QApplication(sys.argv)
from src.gui import Window

window = Window()
window.show()
sys.exit(app.exec_())
