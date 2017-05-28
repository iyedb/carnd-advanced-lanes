import numpy as np
import cv2

import utils


class LaneDetector:
    def __init__(self, nwindows=9, minpixels=50, margin=100):
        self.nwindows = nwindows
        self.minpixels = minpixels
        self.margin = margin
        self.left_fit_poly = None
        self.right_fit_poly = None
        self.minpix = minpixels

    def window_search(self, binary_warped):
        histogram = np.sum(
            binary_warped[binary_warped.shape[0]//2:, :],
            axis=0
        )
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        window_height = np.int(binary_warped.shape[0] / self.nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        left_lane_inds = []
        right_lane_inds = []

        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height

            # left window
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            # right window
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = (
                (nonzeroy >= win_y_low) &
                (nonzeroy < win_y_high) &
                (nonzerox >= win_xleft_low) &
                (nonzerox < win_xleft_high)
            ).nonzero()[0]

            good_right_inds = (
                (nonzeroy >= win_y_low) &
                (nonzeroy < win_y_high) &
                (nonzerox >= win_xright_low) &
                (nonzerox < win_xright_high)
            ).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # update the window center position for the next iteration
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

            if len(good_right_inds) > self.minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        # left lane line pixels
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]

        # right lane line pixels
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        self.left_fit_poly = np.polyfit(lefty, leftx, 2)
        self.right_fit_poly = np.polyfit(righty, rightx, 2)

    def apply_poly(self, arr, poly, margin=0):
        return poly[0]*(arr**2) + poly[1]*arr + poly[2] + margin

    def detect(self, binary_warped):
        if self.left_fit_poly is None and self.right_fit_poly is None:
            self.window_search(binary_warped)

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = (
            (nonzerox >
                self.apply_poly(
                    nonzeroy, self.left_fit_poly, -self.margin
                )) &
            (nonzerox <
                self.apply_poly(
                    nonzeroy, self.left_fit_poly, self.margin))
        )

        right_lane_inds = (
            (nonzerox >
                self.apply_poly(
                    nonzeroy, self.right_fit_poly, -self.margin
                )) &
            (nonzerox <
                self.apply_poly(
                    nonzeroy, self.right_fit_poly, self.margin))
        )

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]

        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        self.left_fit_poly = np.polyfit(lefty, leftx, 2)
        self.right_fit_poly = np.polyfit(righty, rightx, 2)

    def draw_lane_warped(self, binary_warped):
        """
        return a RGB image with a warped green lane drawn
        """
        l, r, ys = self.get_lines(binary_warped)
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))*255
        pts_left = np.array([np.transpose(np.vstack([l, ys]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([r, ys])))])
        pts = np.hstack((pts_left, pts_right))
        return cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    def draw_lane(self, binary_warped, Minv):
        return utils.perspective_change(self.draw_lane_warped(binary_warped), Minv)

    def get_lines(self, binary_warped):
        ploty = np.linspace(
            0,
            binary_warped.shape[0] - 1,
            binary_warped.shape[0]
        )
        left_fitx = self.apply_poly(ploty, self.left_fit_poly)
        right_fitx = self.apply_poly(ploty, self.right_fit_poly)
        return left_fitx, right_fitx, ploty

    def get_curvature(self, binary_warped):
        leftx, rightx, ploty = self.get_lines(binary_warped)
        left_fit = np.polyfit(ploty, leftx, 2)
        right_fit = np.polyfit(ploty, rightx, 2)
        at_y = np.max(ploty)
        left_curverad = (
            (
                1 +
                (2*left_fit[0]*at_y + left_fit[1])**2
            )**1.5
        ) / np.absolute(2*left_fit[0])

        right_curverad = (
            (
                1 +
                (2*right_fit[0]*at_y + right_fit[1])**2
            )**1.5
        ) / np.absolute(2*right_fit[0])
        return left_curverad, right_curverad


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    binary_warped = plt.imread('./test_images/warped_example.jpg')
    detector = LaneDetector()
    detector.detect(binary_warped)
    res = detector.draw_lane_warped(binary_warped)
    print(detector.get_curvature(binary_warped))
    plt.imsave('./output_images/res2.jpg', res)