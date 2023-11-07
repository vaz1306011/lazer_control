"""
單張圖片抓特徵點
"""
import cv2

img = cv2.imread("./image/test.mp4_20230429_012949.513.jpg")

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create(
    threshold=16,
    nonmaxSuppression=True,
    type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16,
)

# find and draw the keypoints
kp = fast.detect(img, None)
img2 = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))

# Print all default params
print("Threshold: {}".format(fast.getThreshold()))
print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
print("neighborhood: {}".format(fast.getType()))
print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))

cv2.imwrite("fast_true.png", img2)

# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img, None)

print("Total Keypoints without nonmaxSuppression: {}".format(len(kp)))

img3 = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))

cv2.imwrite("fast_false.png", img3)
