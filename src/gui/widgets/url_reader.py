from .ui_url_reader import Ui_URLReader
from PyQt5 import QtWidgets


class URLReader(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.ui = Ui_URLReader()
        self.ui.setupUi(self)
        self.handler = None
        self.ui.pushButton.clicked.connect(self.add_url)

    def connect_handler(self, handler):
        self.handler = handler

    def add_url(self):
        if self.handler is None:
            self.close()
            return
        self.handler(self.ui.lineEdit.text())
        self.ui.lineEdit.setText("")
        self.close()
