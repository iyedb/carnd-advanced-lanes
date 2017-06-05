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

    # gradx_binary = abs_sobel_thresh(
    #     undistorted_img,
    #     orient='x',
    #     thresh_min=50,
    #     thresh_max=80,
    #     kernel_size=7
    # )

    luv = cv2.cvtColor(undistorted_img, cv2.COLOR_RGB2LUV)

    lchannel = luv[:, :, 0]
    white_line = np.zeros_like(lchannel)
    white_line[(lchannel >= 210)] = 1

    hls = cv2.cvtColor(undistorted_img, cv2.COLOR_RGB2HLS)

    schannel = hls[:, :, 2]
    sthresh_min = 150
    sthresh_max = 255
    sbinary = np.zeros_like(schannel)
    sbinary[(schannel >= sthresh_min) & (schannel <= sthresh_max)] = 1

    lchannel = hls[:, :, 1]
    lthresh_min = 150
    lthresh_max = 255
    lbinary = np.zeros_like(lchannel)
    lbinary[(lchannel >= lthresh_min) & (lchannel <= lthresh_max)] = 1

    lsbinary = np.zeros_like(schannel)
    lsbinary[(lbinary == 1) & (sbinary == 1)] = 1

    yellow_and_white = np.zeros_like(schannel)
    yellow_and_white[(lsbinary == 1) | (white_line == 1)] = 1

    rgb = np.dstack((lsbinary*255, np.zeros_like(yellow_and_white), white_line*255))
    return yellow_and_white, rgb


def thresh(img, thresh_min, thresh_max):
    ret = np.zeros_like(img)
    ret[(img >= thresh_min) & (img <= thresh_max)] = 1
    return ret


def threshold2(img_array, matrix, coeffs, M, color=False):
    img = np.copy(img_array)
    img = undistort_image(img, matrix, coeffs)
    img = perspective_change(img, M)
    b_img = np.zeros((img.shape[0],img.shape[1]))

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    H = hsv[:,:,0]
    S = hsv[:,:,1]
    V = hsv[:,:,2]

    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]

    t_yellow_H = thresh(H,10,30)
    t_yellow_S = thresh(S,50,255)
    t_yellow_V = thresh(V,150,255)

    t_white_R = thresh(R,225,255)
    t_white_V = thresh(V,230,255)

    yellow_line = np.zeros_like(t_yellow_H)
    white_line = np.zeros_like(t_yellow_H)
    yellow_line[(t_yellow_H==1) & (t_yellow_S==1) & (t_yellow_V==1)] = 1
    white_line[(t_white_R==1)|(t_white_V==1)] = 1

    b_img[(t_yellow_H==1) & (t_yellow_S==1) & (t_yellow_V==1)] = 1
    b_img[(t_white_R==1)|(t_white_V==1)] = 1
    rgb = np.dstack((yellow_line*255, np.zeros_like(yellow_line), white_line*255))

    return b_img, rgb


if __name__ == '__main__':
    import pickle
    camdata = pickle.load(open('./camera_cal/camera_cal_pickle.p', 'rb'))
    mtx = camdata['mtx']
    dist = camdata['dist']

    persp = pickle.load(open('./camera_cal/persp_pickle.p', 'rb'))
    M = persp['M']
    Minv = persp['Minv']

    test_img1 = plt.imread('./test_images/signs_vehicles_xygrad.jpg')
    res, rgb = threshold(test_img1, mtx, dist, M)
    plt.imsave('./output_images/binary1.png', res, cmap='gray')
    plt.imsave('./output_images/color_binary1.png', rgb)

    test_img2 = plt.imread('./extracted_images/frame_42.jpg')
    res, rgb = threshold(test_img2, mtx, dist, M)
    plt.imsave('./output_images/binary21.png', res, cmap='gray')
    plt.imsave('./output_images/color_binary21.png', rgb)

    test_img2 = plt.imread('./extracted_images/frame1299.jpg')
    res, rgb = threshold(test_img2, mtx, dist, M)
    plt.imsave('./output_images/binary31.png', res, cmap='gray')
    plt.imsave('./output_images/color_binary31.png', rgb)
