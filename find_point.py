"""
用顏色和亮度尋找雷射筆
"""
import sys
import time
from dataclasses import dataclass
from enum import Enum
from threading import Timer
from typing import Tuple

import cv2
import keyboard
import numpy as np
import win32api
import win32con
from cv2 import Mat
from PyQt5 import QtCore, QtWidgets

# pyuic5 -x .\buttonUI.ui -o buttonUI.py
from buttonUI import Ui_Dialog

BLACK = [0, 0, 0]
WHITE = [255, 255, 255]
RED = [0, 0, 255]
YELLOW = [0, 255, 255]
GREEN = [0, 255, 0]
BLUE = [255, 0, 0]


class Mode(Enum):
    click = "click"
    doubleClick = "doubleClick"
    drag = "drag"
    drag_up = "drag_up"
    drag_down = "drag_down"


class ButtonUI(QtWidgets.QMainWindow):
    mode_signal = QtCore.pyqtSignal(Mode)

    def __init__(self):
        super().__init__()

        desktop = QtWidgets.QDesktopWidget()
        screen_rect = desktop.screenGeometry()

        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.move((screen_rect.width() - self.width()) // 2, 0)
        self.setFixedSize(self.width(), self.height())
        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.Tool
        )

        self.ui.click.enterEvent = lambda event: self.mode_signal.emit(Mode.click)
        self.ui.doubleClick.enterEvent = lambda event: self.mode_signal.emit(
            Mode.doubleClick
        )
        self.ui.drag.enterEvent = lambda event: self.mode_signal.emit(Mode.drag)


@dataclass
class FourPoints:
    UL: Tuple[int, int] = None
    UR: Tuple[int, int] = None
    BL: Tuple[int, int] = None
    BR: Tuple[int, int] = None

    def __init__(self):
        self._mask_window = QtWidgets.QMainWindow()
        self._mask_window.setStyleSheet(
            "QMainWindow { border: 10px solid red; background-color: white;}"
        )
        self._mask_window.setWindowOpacity(0.75)

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

        # 侵蝕邊緣
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(thresh, kernel, iterations=1)

        # 膨脹邊緣
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
        approx = cv2.approxPolyDP(max_contour, 0.015 * peri, True)

        # 在原始圖像上繪製輪廓和角點
        approx = np.squeeze(approx)
        corners = []
        for point in approx:
            corners.append(tuple(point))

        if len(corners) != 4:
            return self

        return self.set(*corners).sort()


class Trigger(QtCore.QObject):
    task_signal = QtCore.pyqtSignal(Mode)

    def __init__(self) -> None:
        super().__init__()

        self.point: Tuple[int, int] = ()
        self.intervals = []
        self.pre_time = 0
        self._drag_down = False
        self._timer = None

    def press(self):
        if self._timer:
            self._timer.cancel()
        now = time.perf_counter()
        interval = now - self.pre_time
        if interval < 0.3:
            self.intervals.append(interval)
        else:
            self.intervals.clear()
        self.pre_time = now
        print("案", self.point, self.intervals)

        # 拖移-壓
        if len(self.intervals) == 3:

            def drag_down():
                self.task_signal.emit(Mode.drag_down)
                self.intervals.clear()
                self._drag_down = True

            self._timer = Timer(0.3, drag_down)
            self._timer.start()

    def release(self, point: Tuple[int, int], /):
        if not self.intervals:
            print(point)
            self.point = point

        if self._timer:
            self._timer.cancel()
        now = time.perf_counter()
        interval = now - self.pre_time
        if interval < 0.3 and interval:
            self.intervals.append(interval)
        else:
            self.intervals.clear()

        self.pre_time = now
        print("放", self.point, self.intervals)

        # 單擊
        if len(self.intervals) == 2:

            def click():
                self.task_signal.emit(Mode.click)
                self.intervals.clear()

            self._timer = Timer(0.3, click)
            self._timer.start()

        # 雙擊
        if len(self.intervals) == 4:

            def double_click():
                self.task_signal.emit(Mode.doubleClick)
                self.intervals.clear()

            self._timer = Timer(0.3, double_click)
            self._timer.start()

        # 拖移-放
        if self._drag_down:

            def drag_up():
                self.task_signal.emit(Mode.drag_up)
                self._drag_down = False

            self._timer = Timer(0.3, drag_up)
            self._timer.start()


class LazerController:
    # 紅色雷射筆
    red_upper = np.array([180, 255, 255])
    red_lower = np.array([130, 50, 200])

    # 綠色雷射筆
    green_upper = np.array([85, 255, 255])
    green_lower = np.array([35, 37, 200])

    def __init__(self, zoom: float = 1) -> None:
        self._is_running = True

        self._zoom: float = zoom
        self._four_points: FourPoints = FourPoints()
        self._is_mouse_press: bool = False
        self._mode: Mode = Mode.click
        self._point: Tuple[int, int] = ()
        self._pre_point: Tuple[int, int] = ()
        self._trigger = Trigger()
        self._cap = cv2.VideoCapture(0)
        # self._cap = cv2.VideoCapture("./video/test.mkv")
        self._button_ui = ButtonUI()

        self._trigger.task_signal.connect(self._task)
        self._button_ui.mode_signal.connect(self._set_mode)

        keyboard.add_hotkey("esc", self._exit)
        keyboard.add_hotkey("ctrl+f1", lambda: self._four_points.update(self._cap))

    # 當雷射筆按下
    @property
    def on_lazer_press(self) -> bool:
        return (self._point) and (not self._pre_point)

    # 當雷射筆放開
    @property
    def on_lazer_release(self) -> bool:
        return (not self._point) and (self._pre_point)

    # 當滑鼠按下
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

    # 當滑鼠放開
    @property
    def is_mouse_release(self) -> bool:
        return not self.is_mouse_press

    @is_mouse_release.setter
    def is_mouse_release(self, value: bool) -> None:
        self.is_mouse_press = not value

    # 單擊滑鼠
    def click_mouse(self, point) -> None:
        win32api.SetCursorPos(point)
        win32api.mouse_event(
            win32con.MOUSEEVENTF_LEFTDOWN | win32con.MOUSEEVENTF_LEFTUP,
            0,
            0,
        )

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

    def _set_mode(self, mode: Mode) -> None:
        self._mode = mode
        print("set mode", mode)

    def _task(self, mode: Mode) -> None:
        match mode:
            case Mode.click:
                self.click_mouse(self._trigger.point)

            case Mode.doubleClick:
                self.click_mouse(self._trigger.point)
                self.click_mouse(self._trigger.point)

            case Mode.drag_down:
                win32api.SetCursorPos(self._trigger.point)
                self.is_mouse_press = True

            case Mode.drag_up:
                self.is_mouse_release = True

    def _event(self, point: Tuple[int, int]) -> None:
        self._pre_point = self._point
        self._point = point
        if point:
            win32api.SetCursorPos(point)

        if self.on_lazer_press:
            self._trigger.press()

        if self.on_lazer_release:
            self._trigger.release(self._pre_point)

        # match self._mode:
        #     case Mode.click:
        #         if self.on_lazer_release:
        #             self.click_mouse()

        #     case Mode.doubleClick:
        #         if self.on_lazer_release:
        #             self.click_mouse()
        #             self.click_mouse()

        #     case Mode.drag:
        #         if self.on_lazer_release:
        #             self.is_mouse_press = not self.is_mouse_press

        #     case _:
        # raise ValueError("Mode not found")

    def _fliter_point(self) -> None:
        """過濾雷射筆功能"""

        def binary_fliter(gray: Mat) -> Mat:
            _, mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
            # TODO bitwise_and測試
            return cv2.bitwise_and(gray, gray, mask=mask)  # 跟binary_mask做AND

        def hsv_fliter(img: Mat, mask: Mat) -> Mat:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            color_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
            return cv2.bitwise_and(mask, mask, mask=color_mask)  # 跟color_mask做AND

        # 抓雷射筆
        while self._is_running:
            _, img = self._cap.read()

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            binary = binary_fliter(gray)
            hsv = hsv_fliter(img, binary)
            mask = cv2.GaussianBlur(hsv, (13, 13), 0)

            if not self._four_points.is_full():
                continue

            # 投影幕範圍過濾
            filter_area = self._four_points.ndarray

            # 畫投影幕邊框
            cv2.polylines(img, [filter_area], True, YELLOW, 2)

            # 繪製雷射筆邊框
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                # 找面積最大的contour
                contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img, (x, y), (x + w, y + h), RED, 2)
                point = (x + w / 2, y + h / 2)
                converted_point = self._point_convert(point, self._four_points)
                self._event(converted_point)
            else:
                self._event(())

            # mask_img = cv2.bitwise_and(img, img, mask=color_mask)

            # 顯示成果
            show_list = (
                ("img", img),
                # ("mask1", mask1),
                # ("mask2", mask2),
                # ("hsv", hsv),
                # ("color_mask", color_mask),
                # ("mask3", mask),
                # ("four_points_mask", four_points_mask),
                # ("binary", binary),
                # ("mask", mask),
                # ('canny', canny),
                # ('color_mask', color_mask),
                # ("mask_img", mask_img),
                # ("test", test),
            )

            for name, img in show_list:
                cv2.imshow(name, img)
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)

            key = cv2.waitKey(10)
            if key == 27:
                break
            elif key == ord("p"):
                cv2.waitKey(0)

    def set_Pscreen(self) -> None:
        """開啟投影幕範圍選擇視窗"""
        self._four_points.append((0, 0))
        self._four_points.append((0, 1))
        self._four_points.append((1, 0))
        self._four_points.append((1, 1))
        return
        window_name = "set Projection Screen"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_click_handler)

        # 抓取頂點
        while self._is_running:
            _, img = self._cap.read()

            # 繪製梯形頂點
            for point in self._four_points:
                img = cv2.circle(img, point, 5, RED, 2)

            cv2.imshow(window_name, img)

            key = cv2.waitKey(10)
            # backspace
            if key == 8:
                self._four_points.pop()

            # enter
            elif key == 13:
                if len(self._four_points) == 4:
                    break

            # esc
            elif key == 27:
                self._exit()

        self._four_points.sort()
        cv2.destroyWindow(window_name)

    def start(self) -> None:
        self.set_Pscreen()

        # self._button_ui.show()
        self._fliter_point()

        self._cap.release()

    def setting_window(self) -> None:
        """設定視窗"""
        window_name = "setting"
        cv2.namedWindow(window_name)

        nothing = lambda _: _
        cv2.createTrackbar("h", window_name, 0, 255, nothing)
        cv2.createTrackbar("s", window_name, 0, 255, nothing)
        cv2.createTrackbar("v", window_name, 0, 255, nothing)

        h = 0
        s = 0
        v = 0

        cap = cv2.VideoCapture("./video/test.mp4")
        # cap = cv2.VideoCapture(0)
        is_pause = False

        while True:
            if not is_pause:
                _, img = cap.read()

            h = cv2.getTrackbarPos("h", window_name)
            s = cv2.getTrackbarPos("s", window_name)
            v = cv2.getTrackbarPos("v", window_name)

            cv2.imshow(window_name, img)

            key = cv2.waitKey(10)
            if key == 27:
                break
            elif key == ord("p"):
                is_pause = not is_pause

        cv2.destroyWindow(window_name)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    lc = LazerController()
    lc.start()
    # lc._button_ui.show()
    cv2.destroyAllWindows()
    print("done")
    sys.exit(app.exec_())
