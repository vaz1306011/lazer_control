"""
用顏色和 canny 尋找目標
"""
import cv2
import numpy as np
import win32api

BLACK = [0, 0, 0]
WHITE = [255, 255, 255]
RED = [0, 0, 255]
GREEN = [0, 255, 0]
BLUE = [255, 0, 0]

four_points = []


def sort_points(four_points: list):

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


def mouse_click(event, x, y, flags, para):
    if len(four_points) >= 4:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        four_points.append([x, y])


def point_convert(
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

    return (int(width * u), int(height * v))


def main():
    global four_points
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("./video/pos1.MOV")

    # 紅色雷射筆
    # red_upper = np.array([180, 255, 255])
    # red_lower = np.array([130, 50, 200])

    # 綠色雷射筆
    green_upper = np.array([85, 255, 255])
    green_lower = np.array([35, 37, 200])

    cv2.namedWindow("set Projection Screen")
    cv2.setMouseCallback("set Projection Screen", mouse_click)

    # 抓取頂點
    while True:
        _, img = cap.read()

        # 繪製梯形頂點
        for p in four_points:
            img = cv2.circle(img, p, 5, RED, 2)

        cv2.imshow("set Projection Screen", img)

        key = cv2.waitKey(10)
        # backspace
        if key == 8:
            if four_points:
                four_points.pop()

        elif key == 13:
            break

        elif key == 27:
            return

    four_points = sort_points(four_points)
    cv2.destroyAllWindows()

    # 抓雷射筆
    while True:
        # Read the frame
        _, img = cap.read()

        # 灰階
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray

        # 二質化過濾
        _, binary_mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(mask, mask, mask=binary_mask)  # 跟binary_mask做AND

        # HSV過濾
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, green_lower, green_upper)
        mask = cv2.bitwise_and(mask, mask, mask=color_mask)  # 跟color_mask做AND

        # 投影幕範圍過濾
        filter_area = np.array(four_points[:2] + four_points[:1:-1])
        four_points_mask = np.zeros(img.shape, dtype="uint8")
        cv2.fillPoly(four_points_mask, [filter_area], WHITE)
        four_points_mask = cv2.cvtColor(four_points_mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.bitwise_and(mask, mask, mask=four_points_mask)

        # 高斯模糊
        mask = cv2.GaussianBlur(mask, (13, 13), 0)

        cv2.polylines(img, [filter_area], True, RED)

        # 繪製雷射筆邊框
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            cnt = max(contours, key=lambda img: cv2.contourArea(img))
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), GREEN, 2)
            point = (x + w / 2, y + h / 2)
            mouse_pos = point_convert(point, four_points)
            print(mouse_pos)
            win32api.SetCursorPos(mouse_pos)

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

    cap.release()


if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
    print("done")
