import numpy as np
import cv2
import collections
import statistics

import utils

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, maxlen=10):
        
        # x values of the last n fits of the line
        self.recent_xfitted = collections.deque(maxlen=4)

        # average x values of the fitted line over the last n iterations
        self.bestx = None

        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None

        # x values for detected line pixels
        self.all_x = None

        # y values for detected line pixels
        self.all_y = None

        self.recent_fits = collections.deque(maxlen=4)

    def update(self, fit, fit_x, all_x, all_y):
        self.recent_xfitted.append(fit_x)
        self.recent_fits.append(fit)
        self.all_x = all_x
        self.all_y = all_y
        self.bestx = np.average(np.array(list(self.recent_xfitted)), axis=0)
        self.best_fit = np.average(np.array(list(self.recent_fits)), axis=0)

class LaneDetector:
    def __init__(self, nwindows=9, minpixels=50, margin=100):
        self.nwindows = nwindows
        self.minpixels = minpixels
        self.margin = margin
        self.minpix = minpixels
        self.ym_per_pix = 3*15/720
        self.xm_per_pix = 3.7/660.0
        self.curvature = 0.
        self.leftline = Line()
        self.rightline = Line()
        self.skip_window_search = False
        self.fail_count = 0
        self.left_fit = None
        self.right_fit = None
        self.use_last_fit = True
        self.first_frame_processed = False
        self.base_width = []
        self.top_width = []

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
        assert len(leftx) != 0
        assert len(lefty) != 0
        # right lane line pixels
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        assert len(rightx) != 0
        assert len(righty) != 0
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        return leftx, lefty, left_fit, rightx, righty, right_fit

    def apply_poly(self, arr, poly, margin=0):
        return poly[0]*(arr**2) + poly[1]*arr + poly[2] + margin

    def lane_detected(self, left_fitx, right_fitx):
        base_diff = float(np.abs(left_fitx[-1] - right_fitx[-1]))*self.xm_per_pix
        top_diff = float(np.abs(left_fitx[0] - right_fitx[0]))*self.xm_per_pix
        self.base_width.append(base_diff)
        self.top_width.append(top_diff)
        return (
            (base_diff >= 2.5) and
            (base_diff <= 6.0) and
            (top_diff >= 2.5) and
            (top_diff <= 7.0)
        )

    def detect(self, binary_warped):
        ploty = np.linspace(
                0,
                binary_warped.shape[0] - 1,
                binary_warped.shape[0]
        )

        if self.fail_count == 3:
            self.skip_window_search = False
            self.fail_count = 0

        if not self.skip_window_search:
            leftx, lefty, left_fit, rightx, righty, right_fit = self.window_search(binary_warped)
            left_fitx = self.apply_poly(ploty, left_fit)
            right_fitx = self.apply_poly(ploty, right_fit)

            self.left_fit = left_fit
            self.right_fit = right_fit

            if self.lane_detected(left_fitx, right_fitx):
                self.skip_window_search = True
                self.leftline.update(left_fit, left_fitx, leftx, lefty)
                self.rightline.update(right_fit, right_fitx, rightx, righty)
                self.update_curvature(binary_warped.shape, leftx, lefty, rightx, righty, ploty)
                self.use_last_fit = True
                self.first_frame_processed = True
            else:
                if self.first_frame_processed:
                    self.use_last_fit = False
                else:
                    self.use_last_fit = True

        else:
            # a previous lane was detected
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            current_left_fit, current_right_fit = self.get_line_fits()
            left_lane_inds = (
                (nonzerox >
                    self.apply_poly(
                        nonzeroy, current_left_fit, -self.margin
                    )) &
                (nonzerox <
                    self.apply_poly(
                        nonzeroy, current_left_fit, self.margin))
            )

            right_lane_inds = (
                (nonzerox >
                    self.apply_poly(
                        nonzeroy, current_right_fit, -self.margin
                    )) &
                (nonzerox <
                    self.apply_poly(
                        nonzeroy, current_right_fit, self.margin))
            )

            # Again, extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]

            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

            if len(leftx) == 0 or len(lefty == 0) or len(rightx) or len(righty) == 0:
                self.use_last_fit = False
                self.fail_count += 1
                return

            self.left_fit = np.polyfit(lefty, leftx, 2)
            self.right_fit = np.polyfit(righty, rightx, 2)
            left_fitx = self.apply_poly(ploty, self.left_fit)
            right_fitx = self.apply_poly(ploty, self.right_fit)

            if self.lane_detected(left_fitx, right_fitx):
                self.leftline.update2(self.left_fit, left_fitx, leftx, lefty)
                self.rightline.update2(self.right_fit, right_fitx, rightx, righty)
                self.update_curvature(binary_warped.shape, leftx, lefty, rightx, righty, ploty)
                self.use_last_fit = True
                fail_count = 0
            else:
                # detection failed we are going to use the avg from previous good frames
                # and count the times we did this
                self.use_last_fit = False
                self.fail_count += 1

    def get_line_fits(self):
        if self.use_last_fit:
            return self.left_fit, self.right_fit
        return self.leftline.best_fit, self.rightline.best_fit

    def update_curvature(self, shape, leftx, lefty, rightx, righty, ploty):

        left_fit = np.polyfit(lefty*self.ym_per_pix, leftx*self.xm_per_pix, 2)
        right_fit = np.polyfit(righty*self.ym_per_pix, rightx*self.xm_per_pix, 2)
        y_eval = np.max(ploty)
        left_curve_rad = (
            (1 + (2*left_fit[0]*y_eval*self.ym_per_pix + left_fit[1])**2)**1.5
        ) / np.absolute(2*left_fit[0])

        right_curve_rad = (
            (1 + (2*right_fit[0]*y_eval*self.ym_per_pix + right_fit[1])**2)**1.5
        ) / np.absolute(2*right_fit[0])

        self.curvature = np.mean(np.array([left_curve_rad, right_curve_rad]))

    def draw_lane_warped(self, binary_warped):
        """
        return a RGB image with a warped green lane drawn
        """
        l, r, ys = self.get_lines(binary_warped.shape)
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))*255

        pts_left = np.array([np.transpose(np.vstack([l, ys]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([r, ys])))])

        pts = np.hstack((pts_left, pts_right))
        return cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    def draw_lane(self, binary_warped, Minv):
        return utils.perspective_change(self.draw_lane_warped(binary_warped), Minv)

    def get_lines(self, shape):
        left_fit, right_fit = self.get_line_fits()
        ploty = np.linspace(
            0,
            shape[0] - 1,
            shape[0]
        )
        left_fitx = self.apply_poly(ploty, left_fit)
        right_fitx = self.apply_poly(ploty, right_fit)
        return left_fitx, right_fitx, ploty

    def get_position_from_lane_center(self, shape):
        left_fit, right_fit = self.get_line_fits()
        y_eval = shape[0] - 20
        midx = shape[1] / 2
        x_left_pix = left_fit[0]*(y_eval**2) + left_fit[1]*y_eval + left_fit[2]
        x_right_pix = right_fit[0]*(y_eval**2) + right_fit[1]*y_eval + right_fit[2]
        return ((x_left_pix + x_right_pix)/2.0 - midx) * self.xm_per_pix


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    binary_warped = plt.imread('./test_images/warped_example.jpg')
    detector = LaneDetector()
    detector.detect2(binary_warped)
    res = detector.draw_lane_warped(binary_warped)
    print(detector.get_position_from_lane_center(binary_warped.shape))
    print(detector.base_pixels)
    print(detector.top_pixels)
    print(detector.curvature)

    plt.imsave('./output_images/res2.jpg', res)
