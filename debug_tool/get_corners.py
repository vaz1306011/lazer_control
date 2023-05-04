import cv2
import numpy as np

BLACK = [0, 0, 0]
WHITE = [255, 255, 255]
RED = [0, 0, 255]
GREEN = [0, 255, 0]
BLUE = [255, 0, 0]


def main():
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("./video/test.mkv")

    is_pause = False
    while True:
        key = cv2.waitKey(0 if is_pause else 10)
        if key == 27:
            break
        elif key == ord("p"):
            is_pause = not is_pause

        rval, img = cap.read()
        if not rval:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # 自适应阈值化
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        thresh = cv2.bitwise_not(thresh)

        # 侵蝕邊緣
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(thresh, kernel, iterations=1)

        # 边缘检测
        # edges = cv2.Canny(erosion, 100, 200)

        # 膨胀边缘
        kernel = np.ones((9, 9), np.uint8)
        dilation = cv2.dilate(erosion, kernel, iterations=1)

        # 寻找轮廓
        contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 找到面积最大的轮廓
            max_contour = max(contours, key=cv2.contourArea)

            # 找到轮廓的四个角
            peri = cv2.arcLength(max_contour, True)
            approx = cv2.approxPolyDP(max_contour, 0.01 * peri, True)

            cv2.drawContours(img, [approx], -1, RED, 3)

            # 在原始图像上绘制轮廓和角点
            approx = np.squeeze(approx)
            corners = []
            for point in approx:
                corners.append(tuple(point))

        # print(corners)

        # 显示图像
        cv2.namedWindow("image", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow("image", img)
        cv2.imshow("thresh", thresh)
        cv2.imshow("erosion", erosion)
        # cv2.imshow("edges", edges)
        cv2.imshow("dilation", dilation)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
