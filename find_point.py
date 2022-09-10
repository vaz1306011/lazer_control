"""
用顏色和 canny 尋找目標
"""
import cv2
import numpy as np

if __name__ == "__main__":

    BLACK = [0, 0, 0]
    WHITE = [255, 255, 255]
    RED = [0, 0, 255]
    GREEN = [0, 255, 0]
    BLUE = [255, 0, 0]

    cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("./video/pos2.MOV")

    upper = np.array([180, 255, 255])
    lower = np.array([130, 50, 50])

    while True:
        # Read the frame
        rval, img = cap.read()
        if not rval:
            break

        # 轉成灰階
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 高斯模糊
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, 150, 200)

        color_mask = cv2.inRange(hsv, lower, upper)  # 只抓雷射筆顏色
        mask = cv2.bitwise_and(canny, canny, mask=color_mask)  # 跟canny做AND

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            cv2.rectangle(img, (x, y), (x + w, y + h), GREEN, 2)
            # cv2.drawContours(img, [cnt], 0, GREEN, 2)

        # mask_img = cv2.bitwise_and(img, img, mask=color_mask)

        # 顯示成果
        show_list = (
            ("img", img),
            # ('canny', canny),
            # ('color_mask', color_mask),
            # ('mask_img', mask_img),
            # ('mask', mask)
        )

        for name, img in show_list:
            cv2.imshow(name, img)
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)

        key = cv2.waitKey(0)
        if key == 27:
            break
        elif key == ord("p"):
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

    print("done")
