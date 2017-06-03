import cv2
import numpy as np
import matplotlib.pyplot as plt

def perspective_change(img_arr, M):
    return cv2.warpPerspective(
        img_arr,
        M,
        (img_arr.shape[1], img_arr.shape[0])
    )


def undistort_image(img_arr, matrix, coeffs):
    return cv2.undistort(img_arr, matrix, coeffs, None, matrix)


def weighted_img(img, initial_img, α=0.3, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255, kernel_size=3, hls=False):
    # Convert to grayscale
    if hls:
        hls_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        gray = hls_image[:, :, 1]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
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
    img = np.copy(img_array)
    img = undistort_image(img, matrix, coeffs)
    undistorted_img = perspective_change(img, M)

    gradx_binary = abs_sobel_thresh(
        undistorted_img,
        orient='x',
        thresh_min=50,
        thresh_max=80,
        kernel_size=7
    )

    # grady_binary = abs_sobel_thresh(
    #     undistorted_img,
    #     orient='y',
    #     thresh_min=50,
    #     thresh_max=70,
    #     kernel_size=7
    # )

    # mag_binary = magnitude_thresh(
    #     undistorted_img,
    #     sobel_kernel=5,
    #     mag_thresh=(45, 80)
    # )

    # dir_binary = dir_threshold(
    #     undistorted_img,
    #     sobel_kernel=7,
    #     thresh=(.9, 1.3)
    # )

    # combined_mag_dir = np.zeros_like(dir_binary)
    # combined_mag_dir[(mag_binary == 1) & (dir_binary == 1)] = 1

    # combined = np.zeros_like(dir_binary)
    # combined[
    #     (
    #         (gradx_binary == 1) &
    #         (grady_binary == 1)
    #     )
    # ] = 1

    # combined[((gradx_binary == 1) & (grady_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    luv = cv2.cvtColor(undistorted_img, cv2.COLOR_RGB2LUV)
    l_luv = luv[:, :, 0]
    white_line = np.zeros_like(l_luv)
    white_line[(l_luv >= 210)] = 1
    white_line[(gradx_binary == 1) & (white_line == 1)]


    hls = cv2.cvtColor(undistorted_img, cv2.COLOR_RGB2HLS)
    schannel = hls[:, :, 2]
    sthresh_min = 150
    sthresh_max = 255
    sbinary = np.zeros_like(schannel)
    sbinary[(schannel >= sthresh_min) & (schannel <= sthresh_max)] = 1

    hue = hls[:, :, 0]
    hthresh_min = 20
    hthresh_max = 100
    hbinary = np.zeros_like(hue)
    hbinary[(hue >= hthresh_min) & (hue <= hthresh_max)] = 1


    lchannel = hls[:, :, 1]
    lthresh_min = 150
    lthresh_max = 255
    lbinary = np.zeros_like(lchannel)
    lbinary[(lchannel >= lthresh_min) & (lchannel <= lthresh_max)] = 1

    lsbinary = np.zeros_like(hue)
    lsbinary[(lbinary == 1) & (sbinary == 1)] = 1

    color_and_gradient = np.zeros_like(schannel)
    color_and_gradient[(lsbinary == 1) | (white_line == 1)] = 1

    rgb = np.dstack((lsbinary*255, white_line*255, np.zeros_like(color_and_gradient)))
    return color_and_gradient, rgb

if __name__ == '__main__':
    import pickle
    camdata = pickle.load(open('./camera_cal/camera_cal_pickle.p', 'rb'))
    mtx = camdata['mtx']
    dist = camdata['dist']

    persp = pickle.load(open('./camera_cal/persp_pickle.p', 'rb'))
    M = persp['M']
    Minv = persp['Minv']

    # test_img = plt.imread('./test_images/signs_vehicles_xygrad.jpg')
    test_img = plt.imread('./extracted_images/frame_42.jpg')
    res, rgb = threshold(test_img, mtx, dist, M)
    plt.imsave('./output_images/binary.png', res, cmap='gray')
    plt.imsave('./output_images/color_binary.png', rgb)
