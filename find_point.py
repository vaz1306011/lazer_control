"""
用顏色和亮度尋找雷射筆
"""
import sys

# pyuic5 -x .\button.ui -o button.py
import cv2
import numpy as np
import win32api
import win32con
from PyQt5 import QtWidgets

import button as button_ui

BLACK = [0, 0, 0]
WHITE = [255, 255, 255]
RED = [0, 0, 255]
GREEN = [0, 255, 0]
BLUE = [255, 0, 0]


class LazerController:
    # 紅色雷射筆
    red_upper = np.array([180, 255, 255])
    red_lower = np.array([130, 50, 200])

    # 綠色雷射筆
    green_upper = np.array([85, 255, 255])
    green_lower = np.array([35, 37, 200])

    def __init__(self, zoom=1) -> None:
        self.zoom = zoom
        self.four_points = []

    def sort_points(self, four_points: list):

        points = four_points.copy()
        temp = []

        lt = min(points, key=lambda x: x[0] + x[1])
        temp.append(lt)
        points.remove(lt)

        rt = max(points, key=lambda x: x[0] - x[1])
        temp.append(rt)
        points.remove(rt)

        lb = min(points, key=lambda x: x[0] - x[1])
        temp.append(lb)
        points.remove(lb)

        rb = points[0]
        temp.append(rb)
        points.remove(rb)

        return tuple(temp)

    def mouse_click(self, event, x, y, flags, para):
        if len(self.four_points) >= 4:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.four_points.append([x, y])

    def point_convert(
        self,
        point: tuple[int, int],
        four_point: list[list[int]],
        width: int = 1920,
        height: int = 1080,
    ) -> tuple[int, int]:
        x, y = point[0], point[1]
        x0, y0 = four_point[0]
        x1, y1 = four_point[1]
        x2, y2 = four_point[3]
        x3, y3 = four_point[2]

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

        return (int((width * u) / self.zoom), int((height * v) / self.zoom))

    def get_Pscreen(self, cap):
        cv2.namedWindow("set Projection Screen")
        cv2.setMouseCallback("set Projection Screen", self.mouse_click)

        # 抓取頂點
        while True:
            _, img = cap.read()

            # 繪製梯形頂點
            for point in self.four_points:
                img = cv2.circle(img, point, 5, RED, 2)

            cv2.imshow("set Projection Screen", img)

            key = cv2.waitKey(10)
            # backspace
            if key == 8:
                if self.four_points:
                    self.four_points.pop()

            elif key == 13:
                break

            elif key == 27:
                return

        self.four_points = self.sort_points(self.four_points)
        cv2.destroyAllWindows()

    def fliter_point(self, cap):
        has_pre_pos = False
        # 抓雷射筆
        while True:
            # Read the frame
            ret, img = cap.read()

            # 灰階
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask = gray

            # 二質化過濾
            _, binary_mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
            mask = cv2.bitwise_and(mask, mask, mask=binary_mask)  # 跟binary_mask做AND

            # HSV過濾
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            color_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
            mask = cv2.bitwise_and(mask, mask, mask=color_mask)  # 跟color_mask做AND

            # 投影幕範圍過濾
            filter_area = np.array(self.four_points[:2] + self.four_points[:1:-1])
            # four_points_mask = np.zeros(img.shape, dtype="uint8")
            # cv2.fillPoly(four_points_mask, [filter_area], WHITE)
            # four_points_mask = cv2.cvtColor(four_points_mask, cv2.COLOR_BGR2GRAY)
            # mask = cv2.bitwise_and(mask, mask, mask=four_points_mask)

            # 高斯模糊
            mask = cv2.GaussianBlur(mask, (13, 13), 0)

            # 畫投影幕邊框
            cv2.polylines(img, [filter_area], True, RED)

            # 繪製雷射筆邊框
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                cnt = max(contours, key=lambda img: cv2.contourArea(img))
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x + w, y + h), GREEN, 2)
                point = (x + w / 2, y + h / 2)
                mouse_pos = self.point_convert(point, self.four_points)
                # print(mouse_pos)
                win32api.SetCursorPos(mouse_pos)

                has_pre_pos = True

            else:
                if has_pre_pos:
                    win32api.mouse_event(
                        win32con.MOUSEEVENTF_LEFTDOWN | win32con.MOUSEEVENTF_LEFTUP,
                        0,
                        0,
                    )

                has_pre_pos = False

            # mask_img = cv2.bitwise_and(img, img, mask=color_mask)

            # 顯示成果
            show_list = (
                ("img", img),
                # ("four_points_mask", four_points_mask),
                # ("binary", binary_mask),
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

            if key == ord("p"):
                cv2.waitKey(0)

            elif key == 27:
                break

    def start(self):
        cap = cv2.VideoCapture(0)
        # cap = cv2.VideoCapture("./video/pos1.MOV")

        self.get_Pscreen(cap)
        self.fliter_point(cap)

        cap.release()

    def setting_window(self):
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


class ButtonUI(QtWidgets.QMainWindow, button_ui.Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ButtonUI()
    window.show()
    # sys.exit(app.exec_())

    lc = LazerController()
    lc.start()

    cv2.destroyAllWindows()
    print("done")
