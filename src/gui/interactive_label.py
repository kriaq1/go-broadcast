from PyQt5 import QtWidgets, QtGui


class Interactive(QtWidgets.QLabel):
    def __init__(self, *args, **kwargs):
        QtWidgets.QLabel.__init__(self, *args, **kwargs)
        self.handler = None

    def connect_mouse(self, handler):
        self.handler = handler

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self.handler is not None:
            self.handler(ev)
