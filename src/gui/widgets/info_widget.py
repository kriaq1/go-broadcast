from PyQt5 import QtWidgets, QtCore


class InfoWidget(QtWidgets.QTextEdit):
    def __init__(self, text=""):
        QtWidgets.QTextEdit.__init__(self)
        self.resize(800, 600)
        self.setWindowTitle("Info")
        self.setStyleSheet("color:white;")
        self.setMinimumSize(QtCore.QSize(800, 600))
        self.setMaximumSize(QtCore.QSize(800, 600))
        self.setMarkdown(text)
        self.setReadOnly(True)
        self.setStyleSheet("""
        QWidget{
        background-color:rgb(60, 63, 65);
        color:rgb(169, 183, 198);;
        }
        """)
