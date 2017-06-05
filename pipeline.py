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
        self.reset_counter = 0
        self.frame_counter = 0
        self.detection_failures_counter = 0

    def process_frame(self, frame):
        self.frame_counter += 1
        binary_warped, rgb = thresholding.threshold2(
            frame,
            self.mtx,
            self.coeffs,
            self.M
        )

        self.detector.detect3(binary_warped)
        # detected =  self.detector.use_last_fit
        # if not detected:
        #     self.detection_failures_counter += 1
        lane = self.detector.draw_lane(binary_warped, self.Minv)
        frame = utils.weighted_img(frame, lane)

        # if self.curv is None:
        #     self.curv = 'Radius of Curvature: %.2fm' % self.detector.curvature

        self.curv = 'Radius of Curvature: %.2fm' % self.detector.curvature
        cv2.putText(
            frame,
            self.curv,
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 255, 255), 2
        )

        dist = np.absolute(self.detector.get_position_from_lane_center(binary_warped.shape))
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

        # cv2.putText(
        #     frame,
        #     'Detection failure (%d/%d)' % (self.detection_failures_counter, self.frame_counter),
        #     (20,120),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1, (255, 0, 0), 2
        # )

        # cv2.putText(
        #     frame,
        #     'Lane widths: %.2f, %2f' % (self.detector.base_width[-1], self.detector.top_width[-1]),
        #     (20,160),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1, (255, 0, 0), 2
        # )

        # cv2.putText(
        #     frame,
        #     'Lane spacings: base: {}'.format(self.detector.base_pixels),
        #     (20,200),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1, (255, 0, 0), 2
        # )
        # cv2.putText(
        #     frame,
        #     'Lane spacings: top: {}'.format(self.detector.top_pixels),
        #     (20,240),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1, (255, 0, 0), 2
        # )


        # if self.detection_failures_counter == 5:
        #     self.reset_detector()
        #     self.detection_failures_counter = 0

        return self.compose_frame(frame, binary_warped, rgb)

    def __call__(self, frame):
        return self.process_frame(frame)

    def compose_frame(self, frame1, frame2, frame3):
        _output1 = np.dstack((np.zeros_like(frame2), frame2*255, np.zeros_like(frame2)))
        output1 = cv2.resize(_output1,(640, 360), interpolation = cv2.INTER_AREA)
        output2 = cv2.resize(frame3,(640, 360), interpolation = cv2.INTER_AREA)
        outframe = np.zeros((720, 1280+640, 3))
        outframe[:720, :1280,:] = frame1
        outframe[:360, 1280:1920,:] = output1
        outframe[360:720, 1280:1920,:] = output2
        return outframe

    def reset_detector(self):
        self.detector.reset()
        self.reset_counter += 1


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


