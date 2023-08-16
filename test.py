import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import cv2
import keyboard
import numpy as np
import win32api
import win32con
from cv2 import Mat
from PyQt5 import QtCore, QtGui, QtWidgets

# pyuic5 -x .\buttonUI.ui -o buttonUI.py


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setStyleSheet(
            "QMainWindow { border: 10px solid red; background-color: white;}"
        )
        Dialog.setWindowOpacity(0.75)
        # Dialog.showFullScreen()
