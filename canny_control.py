"""
canny 數值可視畫調整
"""
import cv2


def nothing(x):
    pass


if __name__ == "__main__":
    cv2.namedWindow("res")

    cv2.createTrackbar("max", "res", 0, 255, nothing)
    cv2.createTrackbar("min", "res", 0, 255, nothing)

    maxVal = 200
    minVal = 100

    # cap = cv2.VideoCapture('./video/pos2.MOV')
    cap = cv2.VideoCapture(0)
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
