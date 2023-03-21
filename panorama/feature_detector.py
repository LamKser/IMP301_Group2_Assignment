from collections import OrderedDict

import cv2


class FeatureDetector:
    """https://docs.opencv2.org/4.x/d0/d13/classcv2_1_1Feature2D.html"""

    DETECTOR_CHOICES = OrderedDict()

    DETECTOR_CHOICES["orb"] = cv2.ORB.create
    DETECTOR_CHOICES["sift"] = cv2.SIFT_create
    # DETECTOR_CHOICES["brisk"] = cv2.BRISK_create()
    # DETECTOR_CHOICES["akaze"] = cv2.AKAZE_create()

    # DEFAULT_DETECTOR = list(DETECTOR_CHOICES.keys())[0]

    def __init__(self, detector="sift", **kwargs):
        self.detector = FeatureDetector.DETECTOR_CHOICES[detector](**kwargs)

    def detect_features(self, img, *args, **kwargs):
        # print(dir(cv2.detail.computeImageFeatures2(self.detector, img, *args, **kwargs)))
        return cv2.detail.computeImageFeatures2(self.detector, img, *args, **kwargs)

    @staticmethod
    def draw_keypoints(img, features, **kwargs):
        kwargs.setdefault("color", (0, 255, 0))
        keypoints = features.getKeypoints()
        return cv2.drawKeypoints(img, keypoints, None, **kwargs)
