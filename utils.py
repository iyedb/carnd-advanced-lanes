import cv2


def perspective_change(img_arr, M):
    return cv2.warpPerspective(
        img_arr,
        M,
        (img_arr.shape[1], img_arr.shape[0])
    )


def undistort_image(img_arr, matrix, coeffs):
    return cv2.undistort(img_arr, matrix, coeffs, None, matrix)
