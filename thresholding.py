import cv2
import numpy as np
import matplotlib.pyplot as plt

import utils


def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255, kernel_size=3, hls=False):
    # Convert to grayscale
    if hls:
        hls_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        gray = hls_image[:, :, 1]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(abs_sobel/np.max(abs_sobel)*255)
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output


def magnitude_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    gradmag = (gradmag/np.max(gradmag)*255).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def threshold(img_array, matrix, coeffs, M, color=False):
    """
    This function take a image as input and does the following
    - Undistort the image using cam cal data
    - Apply gradient thresholding and color thresholding and combine in a
    binary image
    - warp the image
    return the result
    """
    undistorted_img = utils.undistort_image(img_array, matrix, coeffs)
    gradx_binary = abs_sobel_thresh(
        undistorted_img,
        orient='x',
        thresh_min=30,
        thresh_max=150,
        kernel_size=9
    )

    grady_binary = abs_sobel_thresh(
        undistorted_img,
        orient='y',
        thresh_min=50,
        thresh_max=70,
        kernel_size=9
    )

    mag_binary = magnitude_thresh(
        undistorted_img,
        sobel_kernel=5,
        mag_thresh=(45, 80)
    )

    dir_binary = dir_threshold(
        undistorted_img,
        sobel_kernel=7,
        thresh=(.9, 1.3)
    )

    combined_mag_dir = np.zeros_like(dir_binary)
    combined_mag_dir[(mag_binary == 1) & (dir_binary == 1)] = 1

    combined = np.zeros_like(dir_binary)
    # combined[
    #     (
    #         (gradx_binary == 1) &
    #         (grady_binary == 1)
    #     )
    # ] = 1

    combined[((gradx_binary == 1) & (grady_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1


    hls = cv2.cvtColor(img_array, cv2.COLOR_RGB2HLS)
    schannel = hls[:, :, 2]
    sthresh_min = 130
    sthresh_max = 255
    sbinary = np.zeros_like(schannel)
    sbinary[(schannel >= sthresh_min) & (schannel <= sthresh_max)] = 1

    color_and_gradient = np.zeros_like(schannel)
    color_and_gradient[(sbinary == 1) | (combined == 1)] = 1

    if color:
        return utils.perspective_change(
            np.dstack((sbinary, combined, np.zeros_like(sbinary))),
            M
        )
    return utils.perspective_change(color_and_gradient, M)


if __name__ == '__main__':
    import pickle
    camdata = pickle.load(open('./camera_cal/camera_cal_pickle.p', 'rb'))
    mtx = camdata['mtx']
    dist = camdata['dist']
    persp = pickle.load(open('./camera_cal/persp_pickle.p', 'rb'))
    M = persp['M']
    Minv = persp['Minv']
    test_img = plt.imread('./test_images/test6.jpg')
    res = threshold(test_img, mtx, dist, M)
    plt.imsave('./output_images/binary.jpg', res, cmap='gray')
    res = threshold(test_img, mtx, dist, M, True)
    plt.imsave('./output_images/color_binary.jpg', res)
