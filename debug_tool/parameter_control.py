import sys
from operator import setitem

import cv2
import numpy as np
from cv2 import Mat
from parameterUI import Ui_MainWindow
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication

# pyuic5 -x .\debug_tool\parameterUI.ui -o .\debug_tool\parameterUI.py
MIN_BRIGHTNESS = 230
MAX_BRIGHTNESS = 255
MIN_HSV = np.array([35, 37, 200])
MAX_HSV = np.array([85, 255, 255])


def setMaxBrightness(value):
    global MAX_BRIGHTNESS
    MAX_BRIGHTNESS = value


def setMinBrightness(value):
    global MIN_BRIGHTNESS
    MIN_BRIGHTNESS = value


class Thread(QThread):
    changeOriginalPixmap = pyqtSignal(QImage)
    changeFilterPixmap = pyqtSignal(QImage)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_size = (800, 450)

    def binary_fliter(self, gray: Mat) -> Mat:
        _, mask = cv2.threshold(gray, MIN_BRIGHTNESS, MAX_BRIGHTNESS, cv2.THRESH_BINARY)
        return cv2.bitwise_and(gray, gray, mask=mask)  # 跟binary_mask做AND

    def hsv_fliter(self, img: Mat, mask: Mat) -> Mat:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, MIN_HSV, MAX_HSV)
        return cv2.bitwise_and(mask, mask, mask=color_mask)  # 跟color_mask做AND

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            rval, img = cap.read()
            if not rval:
                break

            # 原影片
            h, w, ch = img.shape
            bytesPerLine = w * ch
            originalImage = QImage(
                img.data, w, h, bytesPerLine, QImage.Format.Format_BGR888
            )
            originalImage = originalImage.scaled(
                self.video_size[0], self.video_size[1], Qt.KeepAspectRatio
            )

            # 過濾後影片
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            binary = self.binary_fliter(gray)
            hsv = self.hsv_fliter(img, binary)
            output = cv2.GaussianBlur(hsv, (13, 13), 0)
            h, w = output.shape
            bytesPerLine = w
            filterImage = QImage(
                output.data, w, h, bytesPerLine, QImage.Format.Format_Grayscale8
            )
            filterImage = filterImage.scaled(
                self.video_size[0], self.video_size[1], Qt.KeepAspectRatio
            )

            self.changeOriginalPixmap.emit(originalImage)
            self.changeFilterPixmap.emit(filterImage)


class App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("過濾調整")
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        th = Thread(self)
        th.changeOriginalPixmap.connect(self.setOriginalImage)
        th.changeFilterPixmap.connect(self.setFilterImage)
        th.start()
        self.show()

        self.ui.minH.setMinimum(0)
        self.ui.minH.setMaximum(255)
        self.ui.minH.setValue(MIN_HSV[0])
        self.ui.minH.valueChanged.connect(lambda val: setitem(MIN_HSV, 0, val))
        self.ui.minS.setMinimum(0)
        self.ui.minS.setMaximum(255)
        self.ui.minS.setValue(MIN_HSV[1])
        self.ui.minS.valueChanged.connect(lambda val: setitem(MIN_HSV, 1, val))
        self.ui.minV.setMinimum(0)
        self.ui.minV.setMaximum(255)
        self.ui.minV.setValue(MIN_HSV[2])
        self.ui.minV.valueChanged.connect(lambda val: setitem(MIN_HSV, 2, val))

        self.ui.maxH.setMinimum(0)
        self.ui.maxH.setMaximum(255)
        self.ui.maxH.setValue(MAX_HSV[0])
        self.ui.maxH.valueChanged.connect(lambda val: setitem(MAX_HSV, 0, val))
        self.ui.maxS.setMinimum(0)
        self.ui.maxS.setMaximum(255)
        self.ui.maxS.setValue(MAX_HSV[1])
        self.ui.maxS.valueChanged.connect(lambda val: setitem(MAX_HSV, 1, val))
        self.ui.maxV.setMinimum(0)
        self.ui.maxV.setMaximum(255)
        self.ui.maxV.setValue(MAX_HSV[2])
        self.ui.maxV.valueChanged.connect(lambda val: setitem(MAX_HSV, 2, val))

        self.ui.minBrigtness.setMinimum(0)
        self.ui.minBrigtness.setMaximum(255)
        self.ui.minBrigtness.setValue(MIN_BRIGHTNESS)
        self.ui.minBrigtness.valueChanged.connect(setMinBrightness)

        self.ui.maxBrigtness.setMinimum(0)
        self.ui.maxBrigtness.setMaximum(255)
        self.ui.maxBrigtness.setValue(MAX_BRIGHTNESS)
        self.ui.maxBrigtness.valueChanged.connect(setMaxBrightness)

    @pyqtSlot(QImage)
    def setOriginalImage(self, image):
        self.ui.original_label.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(QImage)
    def setFilterImage(self, image):
        self.ui.filter_label.setPixmap(QPixmap.fromImage(image))


def on_about_to_quit():
    print("MIN_HSV =", MIN_HSV)
    print("MAX_HSV =", MAX_HSV)
    print("MIN_BRIGHTNESS =", MIN_BRIGHTNESS)
    print("MAX_BRIGHTNESS =", MAX_BRIGHTNESS)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = App()
    app.aboutToQuit.connect(on_about_to_quit)
    sys.exit(app.exec_())
