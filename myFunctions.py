import numpy as np


def myConv2(A, B, param):
    # Get the dimensions of the matrices
    height_A, width_A = A.shape
    height_B, width_B = B.shape

    # Calculate the dimensions of the final matrix based on the param
    if param:
        final_height = height_A
        final_width = width_A
    else:
        final_height = height_A + height_B - 1
        final_width = width_A + width_B - 1

    # Create a final matrix filled with zeros
    final = np.zeros((final_height, final_width))

    # Flip the kernel
    B = np.flip(B)

    # Perform convolution
    for i in range(final_width):  # loop through rows
        for j in range(final_height):  # loop through columns
            for m in range(height_B):  # kernel height loop
                for n in range(width_B):  # kernel width loop
                    if 0 <= i - m < height_A and 0 <= j - n < width_A:
                        final[i, j] += A[i - m, j - n] * B[m, n]

    return final


def myColorToGray(A):
    # Check if the image is already grayscale by seeing if it has only one channel
    if A.ndim == 2 or (A.ndim == 3 and A.shape[2] == 1):
        return A  # If it's already grayscale, return the image as is

    # get copy of the image
    grayscale_image = np.copy(A)

    # Apply the NTSC formula to each pixel in the image by multiplying each of the three channels with the right factor
    grayscale_image = 0.299 * grayscale_image[:, :, 2] + 0.587 * grayscale_image[:, :, 1] + 0.114 * grayscale_image[:,
                                                                                                    :, 0]

    # Convert the resulting grayscale image to uint8
    grayscale_image = grayscale_image.astype(np.uint8)

    return grayscale_image


def myImNoise(A, param, mean=0, sigma=25, split=0.5, amount=0.02):
    # add noise according to the parameters
    # param must be 'gaussian' or 'saltandpepper'

    # Get the copy of the image
    # Create an empty array with the same shape as the image to store the noisy image
    noisy_image = np.copy(A)
    if len(A.shape) == 3:
        rows, cols, channels = noisy_image.shape
        is_color = True
    else:  # grayscale
        rows, cols = noisy_image.shape
        channels = 1
        is_color = False

    if param == 'gaussian':
        for i in range(rows):
            for j in range(cols):
                if is_color:
                    for c in range(channels):
                        # Generate Gaussian noise
                        noise = np.random.normal(mean, sigma)
                        # Add the noise to the original pixel value, ensuring values are valid
                        noisy_pixel = A[i, j, c] + noise
                        # Clip the pixel value to be within valid range
                        noisy_pixel = np.clip(noisy_pixel, 0, 255)
                        # Assign the noisy pixel value to the noisy image
                        noisy_image[i, j, c] = noisy_pixel

                else:
                    # Generate Gaussian noise
                    noise = np.random.normal(mean, sigma)
                    # Add the noise to the original pixel value, ensuring values are valid
                    noisy_pixel = A[i, j] + noise
                    # Clip the pixel value to be within valid range
                    noisy_pixel = np.clip(noisy_pixel, 0, 255)
                    # Assign the noisy pixel value to the noisy image
                    noisy_image[i, j] = noisy_pixel

    elif param == 'saltandpepper':
        # Calculate the number of pixels to effect with salt & pepper noise
        number_of_pixels = int(np.ceil(amount * noisy_image.size * split))
        for i in range(number_of_pixels):
            # Pick random coordinates
            y_coord, x_coord = np.random.randint(0, rows), np.random.randint(0, cols)

            # For color image, affect all channels at the chosen pixel
            if is_color:
                noisy_image[y_coord, x_coord, :] = 255  # White for Salt
            else:
                noisy_image[y_coord, x_coord] = 255

        number_of_pixels = int(np.ceil(amount * noisy_image.size * (1.0 - split)))
        for i in range(number_of_pixels):
            # Pick random coordinates
            y_coord, x_coord = np.random.randint(0, rows), np.random.randint(0, cols)

            # For color image, affect all channels at the chosen pixel
            if is_color:
                noisy_image[y_coord, x_coord, :] = 0  # Black for Pepper
            else:
                noisy_image[y_coord, x_coord] = 0

    else:
        print("ERROR: Wrong param for myImNoise")
        exit()

    return noisy_image


# help function mean filter
def mean_filter(neighborhood):
    # Sum all the elements in the neighborhood
    total_sum = 0
    num_elements = 0
    for row in neighborhood:
        for value in row:
            total_sum += value
            num_elements += 1
    # Calculate the mean
    mean_value = total_sum / num_elements
    return mean_value


# help function for median filter
def quicksort(arr):
    # Base case: if the array is empty or has one element, it's already sorted
    if len(arr) <= 1:
        return arr
    # Choose a pivot element from the array
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    # Recursively apply quicksort to the partitions and combine results
    return quicksort(left) + middle + quicksort(right)


# help function for median filter
def median_filter(neighborhood):
    # Flatten the neighborhood array and sort it using quicksort
    flattened = [item for sublist in neighborhood for item in sublist]
    sorted_values = quicksort(flattened)
    n = len(sorted_values)
    # Find the median
    if n % 2 == 0:
        median_value = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    else:
        median_value = sorted_values[n // 2]

    return median_value


def myImFilter(A, param, kernel_size=3):
    if param != 'mean' and param != 'median':
        print("ERROR: Wrong param for myImFilter")
        exit()

    # Check if the image is color or grayscale
    if len(A.shape) == 3:
        height, width, channels = A.shape
        is_color = True
    else:  # Grayscale image
        height, width = A.shape
        channels = 1  # Treat as if there is one channel
        is_color = False

    # Initialize an output image
    if is_color:
        filtered_image = np.zeros((height, width, channels), dtype=np.uint8)
    else:
        filtered_image = np.zeros((height, width), dtype=np.uint8)

    # Calculate the offset for the kernel
    offset = kernel_size // 2

    # Apply the filter to the image
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            if is_color:
                for c in range(channels):
                    neighborhood = A[y - offset:y + offset + 1, x - offset:x + offset + 1, c]

                    if param == 'mean':
                        value = mean_filter(neighborhood)
                    else:
                        value = median_filter(neighborhood)

                    filtered_image[y, x, c] = value
            else:
                # For grayscale, no need to loop through channels
                neighborhood = A[y - offset:y + offset + 1, x - offset:x + offset + 1]

                if param == 'mean':
                    value = mean_filter(neighborhood)
                else:
                    value = median_filter(neighborhood)

                filtered_image[y, x] = value

    return filtered_image


def myEdgeDetection(A, param):
    # Ensure the image is in grayscale for edge detection
    if len(A.shape) == 3:
        A = myColorToGray(A)
        A = A.astype(np.uint8)  # Ensure it's uint8 for consistency

    if param == 'sobel' or param == 'prewitt':
        if param == 'sobel':
            # Sobel kernels
            kernel_x = np.array([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]])
            kernel_y = np.array([[1, 2, 1],
                                 [0, 0, 0],
                                 [-1, -2, -1]])
        else:  # prewitt kernels
            kernel_x = np.array([[-1, 0, 1],
                                 [-1, 0, 1],
                                 [-1, 0, 1]])
            kernel_y = np.array([[1, 1, 1],
                                 [0, 0, 0],
                                 [-1, -1, -1]])

        # Convolve the image with kernels
        gradient_x = myConv2(A, kernel_x, True)
        gradient_y = myConv2(A, kernel_y, True)

        # Combine gradients
        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

        # Clip and Convert the resulting grayscale image to uint8
        gradient_magnitude = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)

        return gradient_magnitude

    elif param == 'laplacian':
        laplacian_kernel = np.array([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]])
        result = myConv2(A, laplacian_kernel, True)
        result = np.clip(result, 0, 255).astype(np.uint8)  # Clip and Convert the resulting grayscale image to uint8
        return result
