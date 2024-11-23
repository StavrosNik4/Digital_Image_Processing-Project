import cv2
from myFunctions import *

# Load Image
image = cv2.imread("lena.png")

# # Grayscale
A = myColorToGray(image)  # grayscale_image

cv2.imshow("A", A)
cv2.waitKey(0)
cv2.imwrite("./Results/demo1/A.png", A)

# Noise
B = myImNoise(A, 'gaussian')
cv2.imshow("B", B)
cv2.waitKey(0)
cv2.imwrite("./Results/demo1/B.png", B)

# Filter
C = myImFilter(B, 'mean', kernel_size=9)
cv2.imshow("C", C)
cv2.waitKey(0)
cv2.imwrite("./Results/demo1/C.png", C)


# Edge Detection

def printImages(A_edge, E, F, E_edge, F_edge, param):
    cv2.imshow("A_edge " + param, A_edge)
    cv2.waitKey(0)
    cv2.imwrite("./Results/demo1/A_edge_" + param + ".png", A_edge)
    cv2.imshow("E", E)
    cv2.waitKey(0)
    if param == 'sobel':
        cv2.imwrite("./Results/demo1/E.png", E)
    cv2.imshow("F", F)
    cv2.waitKey(0)
    if param == 'sobel':
        cv2.imwrite("./Results/demo1/F.png", F)
    cv2.imshow("E_edge " + param, E_edge)
    cv2.waitKey(0)
    cv2.imwrite("./Results/demo1/E_edge_" + param + ".png", E_edge)
    cv2.imshow("F_edge " + param, F_edge)
    cv2.waitKey(0)
    cv2.imwrite("./Results/demo1/F_edge_" + param + ".png", F_edge)


# Sobel
A_edge = myEdgeDetection(A, 'sobel')
E = myImFilter(A, 'mean', kernel_size=9)
F = myImFilter(A, 'mean', kernel_size=3)
E_edge = myEdgeDetection(E, 'sobel')
F_edge = myEdgeDetection(F, 'sobel')

printImages(A_edge, E, F, E_edge, F_edge, 'sobel')

# Prewitt
A_edge = myEdgeDetection(A, 'prewitt')
E = myImFilter(A, 'mean', kernel_size=9)
F = myImFilter(A, 'mean', kernel_size=3)
E_edge = myEdgeDetection(E, 'prewitt')
F_edge = myEdgeDetection(F, 'prewitt')

printImages(A_edge, E, F, E_edge, F_edge, 'prewitt')

# Laplacian
A_edge = myEdgeDetection(A, 'laplacian')
E = myImFilter(A, 'mean', kernel_size=9)
F = myImFilter(A, 'mean', kernel_size=3)
E_edge = myEdgeDetection(E, 'laplacian')
F_edge = myEdgeDetection(F, 'laplacian')

printImages(A_edge, E, F, E_edge, F_edge, 'laplacian')
