import shutil
import os
import sys
import time
import cv2
import numpy as np
from . import utils
from .ui_mainwindow import Ui_MainWindow
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtMultimedia import QCameraInfo
from ..stream_capture import StreamCapture, StreamSaver, StreamClosed, StreamImage
from .controller import Controller
from .widgets import URLReader, SettingsReader


class Window(QtWidgets.QMainWindow):
    def __init__(self, controller: Controller = None,
                 cache_path=None, global_timestamp=None, fps_save=0.1, fps_update=20,
                 recognition_parameters=None, logging_parameters=None, broadcast_parameters=None):
        QtWidgets.QMainWindow.__init__(self)
        # setup
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.stop_recognition_button.setEnabled(False)
        self.controller = controller
        self.url_writer = URLReader()
        self.recognition_parameters = recognition_parameters
        self.logging_parameters = logging_parameters
        self.broadcast_parameters = broadcast_parameters
        self.recognition_settings = SettingsReader(utils.dict_by_parameters(recognition_parameters), "recognition")
        self.logging_settings = SettingsReader(utils.dict_by_parameters(logging_parameters), "logging")
        self.broadcast_settings = SettingsReader(utils.dict_by_parameters(broadcast_parameters), "broadcast")
        self.global_timestamp = global_timestamp
        self.set_connections()
        # combo box
        self.sources = [None]
        self.current_source = 0
        self.sources_names = [self.ui.choose_source_box.currentText()]
        self.set_available_cameras()
        # stream saver
        self.cache_path = cache_path
        self.fps_save = fps_save
        self.fps_update = fps_update
        # interactive
        self.set_points_flag = False
        # stream image
        self.stream_image = 255 * np.ones((256, 256, 3), dtype=np.uint8)
        self.contour_points = np.array([[-1, -1], [-1, -1], [-1, -1], [-1, -1]])
        self.update_stream_image(self.stream_image)
        # board image
        self.empty_board_image = utils.draw_empty_board()
        self.board_image = self.empty_board_image
        self.last_state = np.ones((19, 19), dtype=int)
        self.last_values = None
        self.update_board_state(self.last_state, self.last_values)
        # stream capture
        self.stream_capture: None | StreamCapture = None
        self.prev_percentage = 100
        # timer
        self.timer = None
        self.set_timer()

    def set_connections(self):
        self.url_writer.connect_handler(self.add_new_source)
        self.recognition_settings.connect_handler(self.update_recognition_parameters)
        self.logging_settings.connect_handler(self.update_logging_parameters)
        self.broadcast_settings.connect_handler(self.update_broadcast_parameters)

        self.ui.start_recognition_button.clicked.connect(self.start_recognition)
        self.ui.stop_recognition_button.clicked.connect(self.stop_recognition)
        self.ui.set_current_state_button.clicked.connect(self.set_current_state)
        self.ui.set_points_button.clicked.connect(self.set_points)
        self.ui.previous_points_box.clicked.connect(self.use_previous_points)
        self.ui.add_source_button.clicked.connect(self.add_source)
        self.ui.add_url_button.clicked.connect(self.add_url)
        self.ui.save_sgf_button.clicked.connect(self.save_sgf)
        self.ui.choose_source_box.activated[int].connect(self.choose_source)
        self.ui.horizontal_slider.valueChanged.connect(self.change_percentage)
        self.ui.actionQuit.triggered.connect(self.quit)
        self.ui.actionRecognition.triggered.connect(self.recognition_parameters_settings)
        self.ui.actionbroadcast.triggered.connect(self.broadcast_parameters_settings)
        self.ui.actionLogging.triggered.connect(self.logging_parameters_settings)
        self.ui.actionInfo.triggered.connect(self.info_widget)

    def quit(self):
        self.close()

    def recognition_parameters_settings(self):
        self.recognition_settings.show()

    def broadcast_parameters_settings(self):
        self.broadcast_settings.show()

    def logging_parameters_settings(self):
        self.logging_settings.show()
        print("logging")

    def info_widget(self):
        print("info widget")

    def update_recognition_parameters(self, parameters):
        self.controller.update_recognition_parameters(**parameters)
        utils.load_dict(parameters, self.recognition_parameters)

    def update_logging_parameters(self, parameters):
        self.controller.update_gamelog_parameters(**parameters)
        utils.load_dict(parameters, self.logging_parameters)

    def update_broadcast_parameters(self, parameters):
        print("broadcast", parameters)

    def mouse_release(self, ev: QtGui.QMouseEvent):
        label_shape = (self.ui.stream_label.height(), self.ui.stream_label.width())
        point = utils.unpadding_points(np.array([ev.x(), ev.y()]), shape=self.stream_image.shape,
                                       padded_shape=label_shape)
        nearest = np.argmin(np.sum((self.contour_points - point) ** 2, axis=1))
        print(self.contour_points, point, label_shape, self.stream_image.shape)
        self.contour_points[nearest] = point

    def set_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.read_stream_capture)
        self.timer.timeout.connect(self.read_board_state)
        self.timer.timeout.connect(self.read_moves)
        self.timer.start(50)

    def set_available_cameras(self):
        cameras = QCameraInfo.availableCameras()
        for camera in cameras:
            name = "{} ({})".format(camera.deviceName(), camera.description())
            self.sources_names.append(name)
            self.ui.choose_source_box.addItem(name)
            self.sources.append(camera.position())

    @pyqtSlot()
    def start_recognition(self):
        if self.controller is not None and self.stream_capture is not None:
            self.controller.update_gamelog_parameters(delay=1)
            self.controller.update_recognition_parameters(source=self.stream_capture.copy())
        self.ui.stop_recognition_button.setEnabled(True)
        self.ui.start_recognition_button.setEnabled(False)
        self.ui.choose_source_box.setEnabled(False)

    @pyqtSlot()
    def stop_recognition(self):
        if self.controller is not None:
            self.controller.update_gamelog_parameters(delay=0)
            self.controller.update_recognition_parameters(source=StreamClosed())
        self.ui.start_recognition_button.setEnabled(True)
        self.ui.choose_source_box.setEnabled(True)
        self.ui.stop_recognition_button.setEnabled(False)

    @pyqtSlot()
    def set_current_state(self):
        if self.stream_capture is None or self.controller is None:
            return
        res, frame, timestamp = self.stream_capture.read()
        if not res:
            return
        self.controller.clear_validation()
        self.controller.update_recognition_parameters(source=StreamImage(frame))
        self.ui.start_recognition_button.setEnabled(True)
        self.ui.choose_source_box.setEnabled(True)
        self.ui.stop_recognition_button.setEnabled(False)

    @pyqtSlot()
    def set_points(self):
        if self.set_points_flag:
            self.ui.set_points_button.setStyleSheet("color: black;")
            self.ui.stream_label.connect_mouse(None)
            if self.controller is not None:
                self.controller.update_recognition_parameters(mode='given',
                                                              points=utils.padding_points(self.contour_points,
                                                                                          self.stream_image.shape))
                self.ui.previous_points_box.setChecked(True)
        else:
            self.ui.set_points_button.setStyleSheet("color: green;")
            self.ui.stream_label.connect_mouse(self.mouse_release)
            self.contour_points = self.controller.last_coordinates(self.stream_image.shape)
        self.set_points_flag = not self.set_points_flag

    @pyqtSlot()
    def use_previous_points(self):
        if self.ui.previous_points_box.isChecked():
            points = self.controller.last_coordinates((1024, 1024))
            self.controller.update_recognition_parameters(mode='given', points=points)
        else:
            self.controller.update_recognition_parameters(mode=None)

    @pyqtSlot()
    def add_source(self):
        file = QtWidgets.QFileDialog.getOpenFileName()[0]
        if not file or file in self.sources:
            return
        self.add_new_source(file)

    def add_new_source(self, source):
        self.ui.choose_source_box.addItem(source)
        self.sources.append(source)

    def add_url(self):
        self.url_writer.show()

    @pyqtSlot()
    def save_sgf(self):
        try:
            file = QtWidgets.QFileDialog.getSaveFileName()[0]
            if file and os.path.exists(self.controller.save_path_sgf):
                shutil.copy(self.controller.save_path_sgf, file)
        except Exception:
            print("Sgf save error", file=sys.stderr)

    @pyqtSlot(int)
    def choose_source(self, id):
        if id == self.current_source:
            return
        self.contour_points = None
        if id == 0:
            self.stream_capture = None
        else:
            del self.stream_capture
            self.controller.update_recognition_parameters(source=StreamClosed())
            self.stream_capture = StreamSaver(self.sources[id], save_path=self.cache_path,
                                              fps_save=self.fps_save, fps_update=self.fps_update,
                                              global_timestamp=self.global_timestamp)

        self.current_source = id

    @pyqtSlot()
    def change_percentage(self):
        pass

    @pyqtSlot()
    def read_stream_capture(self):
        if not self.stream_capture:
            return
        percentage = self.ui.horizontal_slider.value()
        if percentage >= 99:
            ret, image, timestamp = self.stream_capture.read()
        else:
            if percentage == self.prev_percentage:
                return
            timestamp = self.stream_capture.percentage_timestamp(percentage)
            ret, image = self.stream_capture.get(timestamp)
        if ret:
            self.update_stream_image(image)
        self.prev_percentage = percentage

    @pyqtSlot()
    def read_board_state(self):
        if self.controller:
            state, val = self.controller.last_board_state()
            self.update_board_state(state, val)

    @pyqtSlot()
    def read_moves(self):
        pass

    @pyqtSlot(np.ndarray)
    def update_stream_image(self, image):
        self.stream_image = image
        if self.controller:
            if not self.set_points_flag:
                self.contour_points = self.controller.last_coordinates(self.stream_image.shape)
        if self.contour_points is not None:
            self.stream_image = utils.draw_contours(self.stream_image, self.contour_points, thickness=3)
        label_shape = (self.ui.stream_label.height(), self.ui.stream_label.width())
        padded = utils.padding(self.stream_image, shape=label_shape)
        pixmap = utils.convert_cv_qt(padded)
        self.ui.stream_label.setPixmap(pixmap)

    @pyqtSlot(np.ndarray)
    def update_board_image(self, image):
        self.board_image = image
        label_shape = (self.ui.board_label.height(), self.ui.board_label.width())
        padded = utils.padding(image, shape=label_shape)
        pixmap = utils.convert_cv_qt(padded)
        self.ui.board_label.setPixmap(pixmap)

    @pyqtSlot(np.ndarray, np.ndarray)
    def update_board_state(self, state, val=None):
        self.last_state = state
        self.last_values = val
        self.update_board_image(utils.draw_board_state(state, val, self.empty_board_image.copy()))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        self.update_stream_image(self.stream_image)
        self.update_board_image(self.board_image)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        close = QtWidgets.QMessageBox.question(self,
                                               "QUIT",
                                               "Are you sure want to stop process?",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if close == QtWidgets.QMessageBox.Yes:
            event.accept()
            self.logging_settings.close()
            self.broadcast_settings.close()
            self.recognition_settings.close()
            self.url_writer.close()
            del self.stream_capture
            del self.controller
            self.stream_capture = None
            self.controller = None
        else:
            event.ignore()
