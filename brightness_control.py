"""
亮度對比度 數值可視畫調整
"""
import cv2
import numpy as np

# 定義調整亮度對比的函式
def adjust(i, c, b):
    output = i * (c / 100 + 1) - c + b  # 轉換公式
    output = np.clip(output, 0, 255)
    output = np.uint8(output)
    cv2.imshow("img", output)


# 定義調整亮度函式
def brightness_fn(val):
    global img, contrast, brightness
    brightness = val - 100


# 定義調整對比度函式
def contrast_fn(val):
    global img, contrast, brightness
    contrast = val - 100


if __name__ == "__main__":
    cv2.namedWindow("img")

    cv2.createTrackbar("contrast", "img", 0, 200, contrast_fn)
    cv2.setTrackbarPos("contrast", "img", 100)
    cv2.createTrackbar("brightness", "img", 0, 200, brightness_fn)
    cv2.setTrackbarPos("brightness", "img", 100)

    contrast = 0
    brightness = 0

    # cap = cv2.VideoCapture('./video/pos2.MOV')
    cap = cv2.VideoCapture(0)
    is_pause = False

    while True:
        if not is_pause:
            rval, img = cap.read()
            if not rval:
                break

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord("p"):
            is_pause = not is_pause
        adjust(img, contrast, brightness)

    cv2.destroyAllWindows()
