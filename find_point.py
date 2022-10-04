"""
用顏色和 canny 尋找目標
"""
import cv2
import numpy as np


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def point_convert(
    xy: Point, lt: Point, rt: Point, lb: Point, rb: Point, width=1920, height=1080
) -> Point:
    x, y = xy.x, xy.y
    x0, y0 = lt.x, lt.y
    x1, y1 = rt.x, rt.y
    x2, y2 = lb.x, lb.y
    x3, y3 = rb.x, rb.y

    dx1 = x1 - x2
    dx2 = x3 - x2
    dy1 = y1 - y2
    dy2 = y3 - y2
    sx = x0 - x1 + x2 - x3
    sy = y0 - y1 + y2 - y3
    g = (sx * dy2 + dx2 * sy) / (dx1 * dy2 + dx2 * dy1)
    h = (dx1 * sy + sx * dy1) / (dx1 * dy2 + dx2 * dy1)
    a = x1 - x0 + g * x1
    b = x3 - x0 + h * x3
    c = x0
    d = y1 - y0 + g * y1
    e = y3 - y0 + h * y3
    f = y0

    temp = (d * h - e * g) * x + (b * g - a * h) * y + a * e - b * d
    u = ((e - f * h) * x + (c * h - b) * y + b * f - c * e) / temp
    v = ((f * g - d) * x + (a - c * g) * y - c * d - a * f) / temp
    # u = x1 - x0 + g * x1
    # v = y1 - y0 + g * y1

    return Point(width * u, height * v)


if __name__ == "__main__":

    BLACK = [0, 0, 0]
    WHITE = [255, 255, 255]
    RED = [0, 0, 255]
    GREEN = [0, 255, 0]
    BLUE = [255, 0, 0]

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("./video/pos1.MOV")

    upper = np.array([180, 255, 255])
    lower = np.array([130, 50, 200])

    while True:
        # Read the frame
        rval, img = cap.read()
        if not rval:
            break

        # 灰階
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 高斯模糊
        # blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 邊框
        # canny = cv2.Canny(blur, 20, 80)

        # HSV過濾
        color_mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.bitwise_and(gray, gray, mask=color_mask)  # 跟color_mask做AND

        # 繪製邊框
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), GREEN, 2)

        # mask_img = cv2.bitwise_and(img, img, mask=color_mask)

        # 顯示成果
        show_list = (
            ("img", img),
            # ('canny', canny),
            # ('color_mask', color_mask),
            # ('mask_img', mask_img),
            # ('mask', mask)
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

    cap.release()
    cv2.destroyAllWindows()

    print("done")
