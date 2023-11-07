"""
追蹤算法
"""
import cv2

# 打開影片文件
cap = cv2.VideoCapture("video\test2.mkv")

# 選擇一種追蹤算法，例如 MOSSE
tracker = cv2.legacy.TrackerMOSSE_create()

# 讀取第一幀影片
ret, frame = cap.read()

# 選擇初始 ROI（雷射筆光點的位置）
x, y, w, h = cv2.selectROI(frame)
roi = (x, y, w, h)

# 初始化追蹤器
tracker.init(frame, roi)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 更新追蹤器
    ret, roi = tracker.update(frame)

    # 繪製物件位置
    if ret:
        p1 = (int(roi[0]), int(roi[1]))
        p2 = (int(roi[0] + roi[2]), int(roi[1] + roi[3]))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
