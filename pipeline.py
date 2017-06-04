import lanefinder
import thresholding
import utils
import cv2
import random
import numpy as np

class Pipeline:

    def __init__(self, mtx, coeffs, M, Minv):
        self.detector = lanefinder.LaneDetector()
        self.lines = []
        self.mtx = mtx
        self.coeffs = coeffs
        self.M = M
        self.Minv = Minv
        self.curv = None

    def process_frame(self, frame):
        binary_warped = thresholding.threshold(
            frame,
            self.mtx,
            self.coeffs,
            self.M
        )[0]
        self.detector.detect2(binary_warped)
        lane = self.detector.draw_lane(binary_warped, self.Minv)
        frame = utils.weighted_img(frame, lane)
        if self.curv is None:
            self.curv = 'Radius of Curvature: %.2fm' % self.detector.curvature
        # r = random.choice(range(1, 10))
        # if r == 1:
        #     self.curv = 'Radius of Curvature: %.2fm' % self.detector.curvature
        self.curv = 'Radius of Curvature: %.2fm' % self.detector.curvature
        cv2.putText(
            frame,
            self.curv,
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 255, 255), 2
        )

        dist = np.absolute(self.detector.get_position_from_lane_center(binary_warped))
        if dist >= 0:
            dist_text = '%.2fm %s' % (dist, 'right')
        else:
            dist_text = '%.2fm %s' % (dist, 'left')

        cv2.putText(
            frame,
            dist_text,
            (20,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 255, 255), 2
        )

        return frame

    def __call__(self, frame):
        return self.process_frame(frame)


if __name__ == '__main__':
    import pickle
    import matplotlib.pyplot as plt

    camdata = pickle.load(open('./camera_cal/camera_cal_pickle.p', 'rb'))
    mtx = camdata['mtx']
    dist = camdata['dist']
    persp = pickle.load(open('./camera_cal/persp_pickle.p', 'rb'))
    M = persp['M']
    Minv = persp['Minv']

    test_img1 = plt.imread('./test_images/test1.jpg')
    test_img2 = plt.imread('./test_images/signs_vehicles_xygrad.jpg')
    test_img3 = plt.imread('./extracted_images/frame_45.jpg')

    p = Pipeline(mtx, dist, M, Minv)
    res = p.process_frame(test_img1)
    plt.imsave('./output_images/frame1.png', res)

    p = Pipeline(mtx, dist, M, Minv)
    res = p.process_frame(test_img2)
    plt.imsave('./output_images/frame2.png', res)

    p = Pipeline(mtx, dist, M, Minv)
    res = p.process_frame(test_img3)
    plt.imsave('./output_images/frame3.png', res)


