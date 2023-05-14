from PyQt5 import QtWidgets
import sys
from .ui_setting import UiSettings
import os
import yaml
from yaml.loader import SafeLoader


class SettingsReader(QtWidgets.QWidget):
    def __init__(self, parameters=None, name="setting"):
        if type(parameters) == str:
            if os.path.isfile(parameters):
                with open(parameters, "w") as fd:
                    parameters = yaml.load(fd, SafeLoader)
            else:
                parameters = None
        QtWidgets.QWidget.__init__(self)
        self.ui = UiSettings()
        self.setWindowTitle(name)
        if parameters is None:
            parameters = dict()
        self.parameters = parameters
        self.ui.setupUi(self, parameters)
        self.remove_changes()
        self.ui.apply_button.clicked.connect(self.update_parameters)
        self.ui.cancel_button.clicked.connect(self.remove_changes)
        self.handler = None

    def connect_handler(self, handler):
        self.handler = handler

    def update_parameters(self):
        new_parameters = dict()
        try:
            for key in self.ui.labeles.keys():
                new_parameters[key] = type(self.parameters[key])(self.ui.labeles[key][1].text())
        except Exception:
            self.remove_changes()
            print("unsuitable types of values", file=sys.stderr)
            return
        self.parameters = new_parameters
        if self.handler is not None:
            self.handler(self.parameters)

    def remove_changes(self):
        for key in self.ui.labeles.keys():
            self.ui.labeles[key][1].setText(str(self.parameters[key])[0:10])


def write(dct):
    print(dct)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    dct = {"aboba1": 1, "aboba2": 2, "aboba3": 3, "aboba4": 4}
    settings = SettingsReader()
    settings.connect_handler(write)
    settings.show()
    sys.exit(app.exec_())
