import cv2
import cyvlfeat as vl


class FeaturePoint:
    def __init__(self, image):
        self.image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.feature_point = None
        self.descriptor = None

    def detect(self):
        (self.feature_point, self.descriptor) = vl.sift.sift(self.image, peak_thresh=0, edge_thresh=500,
                                                             compute_descriptor=True)
