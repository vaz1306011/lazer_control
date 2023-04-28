"""
用顏色和亮度尋找雷射筆
"""
import sys
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Tuple

import cv2
import keyboard
import numpy as np
import win32api
import win32con
from cv2 import Mat, VideoCapture
from PyQt5 import QtWidgets

# pyuic5 -x .\buttonUI.ui -o buttonUI.py
import buttonUI

BLACK = [0, 0, 0]
WHITE = [255, 255, 255]
RED = [0, 0, 255]
GREEN = [0, 255, 0]
BLUE = [255, 0, 0]


class ButtonUI(QtWidgets.QMainWindow, buttonUI.Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)


class Mode(Enum):
    click = "click"
    doubleClick = "doubleClick"
    drag = "drag"


@dataclass
class FourPoints:
    UL: Tuple[int, int] = None
    UR: Tuple[int, int] = None
    BL: Tuple[int, int] = None
    BR: Tuple[int, int] = None

    def __iter__(self):
        return iter((self.UL, self.UR, self.BL, self.BR))

    def __getitem__(self, index: int):
        return (self.UL, self.UR, self.BL, self.BR)[index]

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

    def update(self, gray: Mat) -> "FourPoints":
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # 自適應二質化
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        thresh = cv2.bitwise_not(thresh)

        # 侵蝕邊緣
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(thresh, kernel, iterations=1)

        # 膨胀边缘
        kernel = np.ones((9, 9), np.uint8)
        dilation = cv2.dilate(erosion, kernel, iterations=1)

        # 找輪廓
        contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return self

        # 找面積最大的輪廓
        max_contour = max(contours, key=cv2.contourArea)

        # 找輪廓的四個角
        peri = cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, 0.01 * peri, True)

        # 在原始图像上绘制轮廓和角点
        approx = np.squeeze(approx)
        corners = []
        for point in approx:
            corners.append(tuple(point))

        if len(corners) != 4:
            return self

        return self.set(*corners).sort()


class LazerController:
    # 紅色雷射筆
    red_upper = np.array([180, 255, 255])
    red_lower = np.array([130, 50, 200])

    # 綠色雷射筆
    green_upper = np.array([85, 255, 225])
    green_lower = np.array([35, 37, 200])

    def __init__(self, zoom: float = 1) -> None:
        self._is_running = True
        keyboard.add_hotkey("esc", self._exit)

        self._zoom = zoom
        self._four_points: FourPoints = FourPoints()
        self._is_mouse_press = False
        self._mode: Mode = Mode.click
        self._point = ()
        self._pre_point = ()

    @property
    def on_lazer_press(self) -> bool:
        return self._point and not self._pre_point

    @property
    def on_lazer_release(self) -> bool:
        return not self._point and self._pre_point

    @property
    def is_mouse_press(self) -> bool:
        return self._is_mouse_press

    @is_mouse_press.setter
    def is_mouse_press(self, value: bool) -> None:
        self._is_mouse_press = value
        if value == False:
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
        else:
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)

    @property
    def is_mouse_release(self) -> bool:
        return not self.is_mouse_press

    @is_mouse_release.setter
    def is_mouse_release(self, value: bool) -> None:
        self.is_mouse_press = not value

    def _exit(self):
        self._is_running = False

    def _mouse_click_handler(self, event, x: int, y: int, flags, para) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._four_points.append((x, y))

    def _point_convert(
        self,
        point: Tuple[int, int],
        four_points: FourPoints,
        width: int = 1920,
        height: int = 1080,
    ) -> Tuple[int, int]:
        """座標正規畫

        Args:
            point (Tuple[int, int]): 相對於整個畫面的點座標
            four_point (FourPoints): 四角點座標
            width (int, optional): 轉換後寬度. Defaults to 1920.
            height (int, optional): 轉換後高度. Defaults to 1080.

        Returns:
            Tuple[int, int]: 正規畫後座標
        """
        x, y = point[0], point[1]
        x0, y0 = four_points[0]
        x1, y1 = four_points[1]
        x2, y2 = four_points[3]
        x3, y3 = four_points[2]

        dx1 = x1 - x2
        dx2 = x3 - x2
        dy1 = y1 - y2
        dy2 = y3 - y2
        sx = x0 - x1 + x2 - x3
        sy = y0 - y1 + y2 - y3
        g = (sx * dy2 - sy * dx2) / (dx1 * dy2 - dy1 * dx2)
        h = (dx1 * sy - dy1 * sx) / (dx1 * dy2 - dy1 * dx2)
        a = x1 - x0 + (g * x1)
        b = x3 - x0 + (h * x3)
        c = x0
        d = y1 - y0 + (g * y1)
        e = y3 - y0 + (h * y3)
        f = y0

        temp = (d * h - e * g) * x + (b * g - a * h) * y + a * e - b * d
        u = ((e - f * h) * x + (c * h - b) * y + b * f - c * e) / temp
        v = ((f * g - d) * x + (a - c * g) * y + c * d - a * f) / temp

        u = max(0, min(u, 1))
        v = max(0, min(v, 1))

        return (int((width * u) / self._zoom), int((height * v) / self._zoom))

    def _set_mode(self, event, mode: Mode) -> None:
        self._mode = mode
        print("set mode", mode)

    def _set_Pscreen(self, cap: VideoCapture) -> None:
        """開啟投影幕範圍選擇視窗

        Args:
            cap (VideoCapture): 攝影機
        """
        cv2.namedWindow("set Projection Screen")
        cv2.setMouseCallback("set Projection Screen", self._mouse_click_handler)

        # 抓取頂點
        while self._is_running:
            _, img = cap.read()

            # 繪製梯形頂點
            for point in self._four_points:
                img = cv2.circle(img, point, 5, RED, 2)

            cv2.imshow("set Projection Screen", img)

            key = cv2.waitKey(10)
            # backspace
            if key == 8:
                self._four_points.pop()

            # enter
            elif key == 13:
                break

            # esc
            elif key == 27:
                self._exit()

        self._four_points.sort()
        cv2.destroyAllWindows()

    def _event(self) -> None:
        match self._mode:
            case Mode.click:
                if self.on_lazer_release:
                    win32api.mouse_event(
                        win32con.MOUSEEVENTF_LEFTDOWN | win32con.MOUSEEVENTF_LEFTUP,
                        0,
                        0,
                    )

            case Mode.doubleClick:
                if self.on_lazer_release:
                    win32api.mouse_event(
                        win32con.MOUSEEVENTF_LEFTDOWN | win32con.MOUSEEVENTF_LEFTUP,
                        0,
                        0,
                    )
                    win32api.mouse_event(
                        win32con.MOUSEEVENTF_LEFTDOWN | win32con.MOUSEEVENTF_LEFTUP,
                        0,
                        0,
                    )

            case Mode.drag:
                if self.on_lazer_release:
                    if not self.is_mouse_press:
                        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
                        self.is_mouse_press = True
                    else:
                        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
                        self.is_mouse_press = False

            case _:
                raise ValueError("Mode not found")

    def _fliter_point(self, cap: VideoCapture) -> None:
        """過濾雷射筆功能

        Args:
            cap (VideoCapture): 攝影機
        """

        def binary_fliter(gray: np.ndarray) -> np.ndarray:
            _, mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
            return mask

        def hsv_fliter(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            color_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
            return cv2.bitwise_and(mask, color_mask)

        def get_corners(gray: np.ndarray) -> FourPoints:
            # 自適應二值化
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # 檢測邊緣
            edges = cv2.Canny(thresh, 100, 200)

            # 邊緣膨脹
            kernel = np.ones((5, 5), np.uint8)
            dilation = cv2.dilate(edges, kernel, iterations=1)

            # 尋找輪廓
            contours, _ = cv2.findContours(
                dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

            # 找到面積最大的輪廓
            max_area = 0
            max_contour = None
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    max_contour = contour

            # 找到輪廓的四個角
            peri = cv2.arcLength(max_contour, True)
            approx = cv2.approxPolyDP(max_contour, 0.02 * peri, True)

            # 取得座標
            approx = np.squeeze(approx)
            corners = []
            for point in approx:
                corners.append(tuple(point))

            print(corners)
            fourpoints = FourPoints(*corners)
            fourpoints.sort()

            return fourpoints

        # 抓雷射筆
        while self._is_running:
            # Read the frame
            _, img = cap.read()

            # 灰階
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 高斯模糊
            blur = cv2.GaussianBlur(gray, (13, 13), 0)

            # 二質化
            binary = binary_fliter(blur)

            # HSV過濾
            mask = hsv_fliter(img, binary)

            # 抓取投影幕四個頂點
            self._four_points.update(gray)

            if not self._four_points.is_full():
                continue

            # 投影幕範圍過濾
            filter_area = np.array(
                [
                    self._four_points.UL,
                    self._four_points.UR,
                    self._four_points.BR,
                    self._four_points.BL,
                ]
            )

            # 畫投影幕邊框
            cv2.polylines(img, [filter_area], True, GREEN)

            # 繪製雷射筆邊框
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            self._pre_point = self._point
            if contours:
                # 找面積最大的contour
                contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img, (x, y), (x + w, y + h), RED, 2)
                self._point = (x + w / 2, y + h / 2)

                converted_point = self._point_convert(self._point, self._four_points)
                # print(mouse_pos)
                win32api.SetCursorPos(converted_point)

            else:
                self._point = ()

            self._event()

            # mask_img = cv2.bitwise_and(img, img, mask=color_mask)

            # 顯示成果
            show_list = (
                ("img", img),
                # ("four_points_mask", four_points_mask),
                # ("binary", binary_mask),
                ("mask", mask),
                # ('canny', canny),
                # ('color_mask', color_mask),
                # ("mask_img", mask_img),
                # ("test", test),
            )

            for name, img in show_list:
                cv2.imshow(name, img)
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)

            key = cv2.waitKey(10)

            if key == ord("p"):
                cv2.waitKey(0)

            elif key == 27:
                break

    def start(self) -> None:
        app = QtWidgets.QApplication(sys.argv)

        button_ui = ButtonUI()
        button_ui.click.enterEvent = partial(self._set_mode, mode=Mode.click)
        button_ui.doubleClick.enterEvent = partial(
            self._set_mode, mode=Mode.doubleClick
        )
        button_ui.drag.enterEvent = partial(self._set_mode, mode=Mode.drag)

        # cap = cv2.VideoCapture(0)
        cap = cv2.VideoCapture("./video/test.mkv")

        # self._set_Pscreen(cap)

        button_ui.show()
        self._fliter_point(cap)

        cap.release()

    def setting_window(self) -> None:
        """設定視窗"""
        cv2.namedWindow("res")

        nothing = lambda _: _
        cv2.createTrackbar("r", "res", 0, 255, nothing)
        cv2.createTrackbar("g", "res", 0, 255, nothing)
        cv2.createTrackbar("b", "res", 0, 255, nothing)

        r = 0
        g = 0
        b = 0

        cap = cv2.VideoCapture("./video/pos2.MOV")
        # cap = cv2.VideoCapture(0)
        is_pause = False

        while True:
            if not is_pause:
                rval, img = cap.read()
                if not rval:
                    break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            key = cv2.waitKey(10)
            if key == 27:
                break
            elif key == ord("p"):
                is_pause = not is_pause

            maxVal = cv2.getTrackbarPos("min", "res")
            minVal = cv2.getTrackbarPos("max", "res")
            if minVal < maxVal:
                edge = cv2.Canny(blur, 100, 200)
                cv2.imshow("res", edge)
            else:
                edge = cv2.Canny(blur, minVal, maxVal)
                cv2.imshow("res", edge)

        print(minVal, maxVal)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    lc = LazerController()
    lc.start()

    cv2.destroyAllWindows()
    print("done")
