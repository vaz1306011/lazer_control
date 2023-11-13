"""
背景減法器
"""
from dataclasses import dataclass
from typing import Tuple

import cv2
import keyboard
import numpy as np
from cv2 import Mat
from PyQt5 import QtWidgets

YELLOW = [0, 255, 255]
WHITE = [255, 255, 255]


@dataclass
class FourPoints:
    UL: Tuple[int, int] = (0, 0)
    UR: Tuple[int, int] = (0, 1)
    BL: Tuple[int, int] = (1, 0)
    BR: Tuple[int, int] = (1, 1)

    def __init__(self):
        keyboard.add_hotkey("ctrl+f1", lambda: self.update(cap))
        # self._mask_window = QtWidgets.QMainWindow()
        # self._mask_window.setStyleSheet(
        #     "QMainWindow { border: 10px solid red; background-color: white;}"
        # )
        # self._mask_window.setWindowOpacity(0.75)

    def __iter__(self):
        return iter((self.UL, self.UR, self.BL, self.BR))

    def __getitem__(self, index: int):
        return (self.UL, self.UR, self.BL, self.BR)[index]

    def __len__(self) -> int:
        return 4 - (self.UL, self.UR, self.BL, self.BR).count(None)

    def __str__(self) -> str:
        return f"UL: {self.UL}, UR: {self.UR}, BL: {self.BL}, BR: {self.BR}"

    @property
    def ndarray(self) -> np.ndarray:
        return np.array([self.UL, self.UR, self.BL, self.BR])

    def is_full(self) -> bool:
        return all((x is not None for x in self))

    def set(self, UL: int, UR: int, BL: int, BR: int) -> "FourPoints":
        self.UL = UL
        self.UR = UR
        self.BL = BL
        self.BR = BR
        return self

    def append(self, point: Tuple[int, int]) -> "FourPoints":
        temp = list(self)
        if None not in temp:
            return

        for i in range(4):
            if temp[i] is None:
                temp[i] = point
                break

        return self.set(*temp)

    def pop(self) -> "FourPoints":
        temp = list(self)
        if not any((x is not None for x in temp)):
            return None

        for i in range(3, -1, -1):
            if temp[i] is not None:
                temp[i] = None
                break

        return self.set(*temp)

    def sort(self) -> "FourPoints":
        """排序四點(按照左上 右上 左下 右下)


        Returns:
            FourPoints: self
        """
        if None in self:
            return self

        y = sorted(self, key=lambda x: x[1])
        U = set(y[:2])
        B = set(y[2:])
        x = sorted(self, key=lambda x: x[0])
        L = set(x[:2])
        R = set(x[2:])

        try:
            UL = (U & L).pop()
            UR = (U & R).pop()
            BL = (B & L).pop()
            BR = (B & R).pop()
        except KeyError:
            return

        return self.set(UL, UR, BL, BR)

    def update(self, cap: Mat) -> "FourPoints":
        # self._mask_window.showFullScreen()
        # time.sleep(0.5)
        _, img = cap.read()
        # self._mask_window.hide()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # 自適應二質化
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        thresh = cv2.bitwise_not(thresh)

        # # 侵蝕邊緣
        # kernel = np.ones((3, 3), np.uint8)
        # erosion = cv2.erode(thresh, kernel, iterations=1)

        # # 膨脹邊緣
        # kernel = np.ones((9, 9), np.uint8)
        # dilation = cv2.dilate(erosion, kernel, iterations=1)

        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        # 找輪廓
        contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return self

        # 找面積最大的輪廓
        max_contour = max(contours, key=cv2.contourArea)

        # 找輪廓的四個角
        peri = cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, 0.015 * peri, True)

        # 在原始圖像上繪製輪廓和角點
        approx = np.squeeze(approx)
        corners = []
        for point in approx:
            corners.append(tuple(point))

        if len(corners) != 4:
            return self

        return self.set(*corners).sort()


backSub = cv2.createBackgroundSubtractorKNN()
# backSub = cv2.createBackgroundSubtractorMOG2()
cap = cv2.VideoCapture("./video/test2.mkv")
fp = FourPoints()

while True:
    ret, frame = cap.read()
    if frame is None:
        break

    fgMask = backSub.apply(frame)

    # 投影幕範圍過濾
    filter_area = fp.ndarray

    four_points_mask = np.zeros(fgMask.shape, dtype="uint8")
    cv2.fillPoly(four_points_mask, [filter_area], WHITE)
    mask = cv2.bitwise_and(mask, four_points_mask)

    # 畫投影幕邊框
    cv2.polylines(frame, [filter_area], True, YELLOW, 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("FG Mask", fgMask)

    keyboard = cv2.waitKey(30)
    if keyboard == "q" or keyboard == 27:
        break
