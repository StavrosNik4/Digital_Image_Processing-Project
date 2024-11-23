import cv2
from myFunctions import *
import numpy as np

# 2D Convolution Example
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

B = np.array([[0, 1, 0],
              [1, 0, 1],
              [0, 1, 0]])

C = myConv2(A, B, False)

print("Result of 2D Convultion Example:")
print(C)

# Load Image
original_image = cv2.imread("lena.png")
cv2.imshow("original_image", original_image)
cv2.waitKey(0)

""" GRAYSCALE """

grayscale_image = myColorToGray(original_image)
cv2.imshow("grayscale_image", grayscale_image)
cv2.waitKey(0)
cv2.imwrite("./Results/demo2/grayscale.png", grayscale_image)

""" GAUSSIAN NOISE """

gaussian_noise_image_mean_0_sigma_25 = myImNoise(original_image, 'gaussian', mean=0, sigma=25)
cv2.imshow("gaussian_noise_image_mean_0_sigma_25", gaussian_noise_image_mean_0_sigma_25)
cv2.waitKey(0)
cv2.imwrite("./Results/demo2/gaussian_noise_image_mean_0_sigma_25.png", gaussian_noise_image_mean_0_sigma_25)

gaussian_noise_image_mean_0_sigma_35 = myImNoise(original_image, 'gaussian', mean=0, sigma=35)
cv2.imshow("gaussian_noise_image_mean_0_sigma_35", gaussian_noise_image_mean_0_sigma_35)
cv2.waitKey(0)
cv2.imwrite("./Results/demo2/gaussian_noise_image_mean_0_sigma_35.png", gaussian_noise_image_mean_0_sigma_35)

gaussian_noise_image_mean_0_sigma_50 = myImNoise(original_image, 'gaussian', mean=0, sigma=50)
cv2.imshow("gaussian_noise_image_mean_0_sigma_50", gaussian_noise_image_mean_0_sigma_50)
cv2.waitKey(0)
cv2.imwrite("./Results/demo2/gaussian_noise_image_mean_0_sigma_50.png", gaussian_noise_image_mean_0_sigma_50)


""" SALT AND PEPPER NOISE """

salt_and_pepper_noise_image_amount_002_split_50 = myImNoise(original_image, 'saltandpepper', amount=0.02, split=0.5)
cv2.imshow("salt_and_pepper_noise_image_amount_0.02_split_50", salt_and_pepper_noise_image_amount_002_split_50)
cv2.waitKey(0)
cv2.imwrite("./Results/demo2/salt_and_pepper_noise_image_amount_0.02_split_50.png", salt_and_pepper_noise_image_amount_002_split_50)

salt_and_pepper_noise_image_amount_005_split_70 = myImNoise(original_image, 'saltandpepper', amount=0.05, split=0.7)
cv2.imshow("salt_and_pepper_noise_image_amount_0.05_split_70", salt_and_pepper_noise_image_amount_005_split_70)
cv2.waitKey(0)
cv2.imwrite("./Results/demo2/salt_and_pepper_noise_image_amount_0.05_split_70.png", salt_and_pepper_noise_image_amount_005_split_70)

salt_and_pepper_noise_image_amount_010_split_30 = myImNoise(original_image, 'saltandpepper', amount=0.1, split=0.3)
cv2.imshow("salt_and_pepper_noise_image_amount_0.10_split_30", salt_and_pepper_noise_image_amount_010_split_30)
cv2.waitKey(0)
cv2.imwrite("./Results/demo2/salt_and_pepper_noise_image_amount_0.10_split_30.png", salt_and_pepper_noise_image_amount_010_split_30)

""" MEAN FILTER """

mean_filtered_image_kernel_5 = myImFilter(original_image, 'mean', kernel_size=5)
cv2.imshow("mean_filtered_image_kernel_5", mean_filtered_image_kernel_5)
cv2.waitKey(0)
cv2.imwrite("./Results/demo2/mean_filtered_image_kernel_5.png", mean_filtered_image_kernel_5)

mean_filtered_image_kernel_7 = myImFilter(original_image, 'mean', kernel_size=7)
cv2.imshow("mean_filtered_image_kernel_7", mean_filtered_image_kernel_7)
cv2.waitKey(0)
cv2.imwrite("./Results/demo2/mean_filtered_image_kernel_7.png", mean_filtered_image_kernel_7)

mean_filtered_image_kernel_9 = myImFilter(original_image, 'mean', kernel_size=9)
cv2.imshow("mean_filtered_image_kernel_9", mean_filtered_image_kernel_9)
cv2.waitKey(0)
cv2.imwrite("./Results/demo2/mean_filtered_image_kernel_9.png", mean_filtered_image_kernel_9)

""" MEDIAN FILTER """

median_filtered_image_kernel_5 = myImFilter(original_image, 'median', kernel_size=5)
cv2.imshow("median_filtered_image_kernel_5", median_filtered_image_kernel_5)
cv2.waitKey(0)
cv2.imwrite("./Results/demo2/median_filtered_image_kernel_5_a.png", median_filtered_image_kernel_5)

median_filtered_image_kernel_7 = myImFilter(original_image, 'median', kernel_size=7)
cv2.imshow("median_filtered_image_kernel_7", median_filtered_image_kernel_7)
cv2.waitKey(0)
cv2.imwrite("./Results/demo2/median_filtered_image_kernel_7.png", median_filtered_image_kernel_7)

median_filtered_image_kernel_9 = myImFilter(original_image, 'median', kernel_size=9)
cv2.imshow("median_filtered_image_kernel_9", median_filtered_image_kernel_9)
cv2.waitKey(0)
cv2.imwrite("./Results/demo2/median_filtered_image_kernel_9.png", median_filtered_image_kernel_9)

""" EDGE DETECTION """

# Sobel
sobel_image = myEdgeDetection(grayscale_image, 'sobel')
cv2.imshow("sobel_image", sobel_image)
cv2.waitKey(0)
cv2.imwrite("./Results/demo2/sobel_image.png", sobel_image)

# Prewitt
prewitt_image = myEdgeDetection(grayscale_image, 'prewitt')
cv2.imshow("prewitt_image", prewitt_image)
cv2.waitKey(0)
cv2.imwrite("./Results/demo2/prewitt_image.png", prewitt_image)

# Laplacian
laplacian_image = myEdgeDetection(grayscale_image, 'laplacian')
cv2.imshow("laplacian_image", laplacian_image)
cv2.waitKey(0)
cv2.imwrite("./Results/demo2/laplacian_image.png", laplacian_image)