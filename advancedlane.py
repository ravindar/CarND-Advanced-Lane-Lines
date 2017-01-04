import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

## Camera Calibration

#reading in an image
image = mpimg.imread('camera_cal/calibration1.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(image)  #call as plt.imshow(gray) to show a image

images = glob.glob('camera_cal/calibration*.jpg')

objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image place

objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

for fname in images:
    img = mpimg.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

## Apply the distortion correction to the raw image
image = mpimg.imread('camera_cal/calibration1.jpg')
undst = cv2.undistort(image, mtx, dist, None, mtx)

f, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 10))
ax1.imshow(image)
ax1.set_title('Original', fontsize=15)
ax2.imshow(undst)
ax2.set_title('Undistorted', fontsize=15)

## Perspective transform
def perspective_transform(undist_image, src, dst, img_size):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(undist_image, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

image = mpimg.imread('test_images/test1.jpg')
undst = cv2.undistort(image, mtx, dist, None, mtx)
src =  np.float32([[240,700],[1130,700],[600,450],[750,450]])
dst = np.float32([[200,700],[1130,700],[200,50],[1130,50]])

f, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(12, 15))

#trans_image = cv2.fillConvexPoly(undst, np.array([[270,700],[1130,700],[800,500],[500,520]]), (0,255,0) )

ax1.imshow(undst)
ax1.plot(240, 700, ".")
ax1.plot(1130, 700, ".")
ax1.plot(600, 450, ".")
ax1.plot(750, 450, ".")
ax1.set_title('Original', fontsize=15)

transformed = perspective_transform(undst, src, dst, (image.shape[1], image.shape[0]))
ax2.imshow(transformed)
ax2.plot(200, 700, ".")
ax2.plot(1130, 700, ".")
ax2.plot(200, 50, ".")
ax2.plot(1130, 50, ".")
ax2.set_title('Transformed', fontsize=15)

image = mpimg.imread('test_images/test2.jpg')
undst = cv2.undistort(image, mtx, dist, None, mtx)
ax3.imshow(undst)
ax3.plot(240, 700, ".")
ax3.plot(1130, 700, ".")
ax3.plot(600, 450, ".")
ax3.plot(750, 450, ".")
ax3.set_title('Original', fontsize=15)

transformed = perspective_transform(undst, src, dst, (image.shape[1], image.shape[0]))
ax4.imshow(transformed)
ax4.plot(200, 700, ".")
ax4.plot(1130, 700, ".")
ax4.plot(200, 50, ".")
ax4.plot(1130, 50, ".")
ax4.set_title('Transformed', fontsize=15)

image = mpimg.imread('test_images/test3.jpg')
undst = cv2.undistort(image, mtx, dist, None, mtx)
ax5.imshow(undst)
ax5.plot(240, 700, ".")
ax5.plot(1130, 700, ".")
ax5.plot(600, 450, ".")
ax5.plot(750, 450, ".")
ax5.set_title('Original', fontsize=15)

transformed = perspective_transform(undst, src, dst, (image.shape[1], image.shape[0]))
ax6.imshow(transformed)
ax6.plot(200, 700, ".")
ax6.plot(1130, 700, ".")
ax6.plot(200, 50, ".")
ax6.plot(1130, 50, ".")
ax6.set_title('Transformed', fontsize=15)

image = mpimg.imread('test_images/test4.jpg')
undst = cv2.undistort(image, mtx, dist, None, mtx)
ax7.imshow(undst)
ax7.plot(240, 700, ".")
ax7.plot(1130, 700, ".")
ax7.plot(600, 450, ".")
ax7.plot(750, 450, ".")
ax7.set_title('Original', fontsize=15)

transformed = perspective_transform(undst, src, dst, (image.shape[1], image.shape[0]))
ax8.imshow(transformed)
ax8.plot(200, 700, ".")
ax8.plot(1130, 700, ".")
ax8.plot(200, 50, ".")
ax8.plot(1130, 50, ".")
ax8.set_title('Transformed', fontsize=15)

## Use color transforms, gradients, etc., to create a thresholded binary image and then do a perspective transform
def color_transform(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return s_binary, combined_binary


image = mpimg.imread('test_images/test4.jpg')
undst = cv2.undistort(image, mtx, dist, None, mtx)

s_b, combined_b = color_transform(image)

s_b_transformed = perspective_transform(s_b, src, dst, (s_b.shape[1], s_b.shape[0]))
combined_b_transformed = perspective_transform(combined_b, src, dst, (combined_b.shape[1], combined_b.shape[0]))

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
ax1.set_title('s channel thresholded')
ax1.imshow(s_b)

ax2.set_title('Combined S channel and gradient thresholds')
ax2.imshow(combined_b, cmap='gray')

ax3.set_title('s channel thresholded (perspective transformed)')
ax3.imshow(s_b_transformed)

ax4.set_title('Combined S channel and gradient thresholds (perspective transformed)')
ax4.imshow(combined_b_transformed, cmap='gray')

#Detect lane pixels and fit to find lane boundary.

import scipy.signal

xcord_left_line = []
ycord_left_line = []
xcord_right_line = []
ycord_right_line = []

size = combined_b_transformed.shape[0]//10
for i in range(0, 10):
    start = i * combined_b_transformed.shape[0]//10
    end = (i + 1) * combined_b_transformed.shape[0]//10
    histogram = np.sum(combined_b_transformed[start:end,:], axis=0)
    peaks = np.array(scipy.signal.find_peaks_cwt(histogram, np.array([200])))

    if len(peaks) == 0:
        peaks = None
        continue

    if len(peaks) == 1:
        xcord_left_line.append(peaks[0])
        ycord_left_line.append(histogram[peaks[0]].astype(int))
    else:
        xcord_left_line.append(peaks[0])
        ycord_left_line.append(histogram[peaks[0]].astype(int))
        xcord_right_line.append(peaks[1])
        ycord_right_line.append(histogram[peaks[1]].astype(int))

ycord_left_line = np.array(ycord_left_line)
xcord_left_line = np.array(xcord_left_line)
ycord_left_line = ycord_left_line[::-1]
xcord_left_line = xcord_left_line[::-1]

ycord_right_line = np.array(ycord_right_line)
xcord_right_line = np.array(xcord_right_line)
ycord_right_line = ycord_right_line[::-1]
xcord_right_line = xcord_right_line[::-1]

# for i in reversed(range(10, 100)):
#     histogram = np.sum(combined_b_transformed[
#                        i * combined_b_transformed.shape[0] / 100:(i + 1) * combined_b_transformed.shape[0] / 100, :],
#                        axis=0)
#     # xcord is 248, our starting point for src is 240 and ending is 1130, so adding 1130-240-248 will give us right line
#     xcord_left_line.append(int(np.argmax(histogram)))
#     ycord_left_line.append(int(i * combined_b_transformed.shape[0] / 100))
#     xcord_right_line.append(int(np.argmax(histogram)) + 642)
#     ycord_right_line.append(int(i * combined_b_transformed.shape[0] / 100))

# Fit a second order polynomial to each lane line
left_fit = np.polyfit(np.array(ycord_left_line), np.array(xcord_left_line), 2)
left_fitx = left_fit[0] * np.array(ycord_left_line) ** 2 + left_fit[1] * np.array(ycord_left_line) + left_fit[2]

right_fit = np.polyfit(np.array(ycord_right_line), np.array(xcord_right_line), 2)
right_fitx = right_fit[0] * np.array(ycord_right_line) ** 2 + right_fit[1] * np.array(ycord_right_line) + right_fit[2]

# Plot up the fake data
plt.imshow(combined_b_transformed)
plt.xlim(0, 1280)
plt.ylim(0, 720)
plt.plot(np.array(reversed(left_fitx)), np.array(ycord_left_line), color='green', linewidth=3)
plt.plot(np.array(reversed(right_fitx)), np.array(ycord_right_line), color='green', linewidth=3)
plt.gca().invert_yaxis()


