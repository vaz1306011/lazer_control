"""
單張圖片測試
"""
import cv2
import numpy as np


def draw_min_rect_circle(img, cnts):  # conts = contours
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # blue


if __name__ == "__main__":
    upper = np.array([255, 255, 255])
    lower = np.array([200, 200, 215])

    img = cv2.imread("./image/test.png")

    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    canny = cv2.Canny(blur, 100, 150)

    mask = cv2.inRange(img, lower, upper)
    out = cv2.bitwise_and(img, img, mask=mask)

    # contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # draw_min_rect_circle(img, contours)

    cv2.imshow("img", img)  # 秀出圖片
    cv2.imshow("mask", mask)  # 秀出圖片
    cv2.imshow("out", out)  # 秀出圖片
    cv2.waitKey(0)
    cv2.destroyAllWindows()
