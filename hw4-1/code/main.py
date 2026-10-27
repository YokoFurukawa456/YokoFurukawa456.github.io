import numpy as np
import cv2
import os

#A.1: Shoot the Pictures
#import images
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

image1 = cv2.imread(os.path.join(DATA_DIR, "image1.JPG"))
image2 = cv2.imread(os.path.join(DATA_DIR, "image2.JPG"))
image1 = cv2.resize(image1, None, fx=0.25, fy=0.25)
image2 = cv2.resize(image2, None, fx=0.25, fy=0.25)

assert image1 is not None and image2 is not None, "couldn't load images"

#A.2: Recover Homographies
#recover homography
#H = computeH(im1_pts,im2_pts)

def computeH(image1_pts, image_pts):
    # Compute the homography matrix H that maps points from image1 to image2
    # image1_pts and image2_pts are Nx2 arrays of corresponding points
    A = []
    for i in range(len(image1_pts)):
        x, y = image1_pts[i][0], image1_pts[i][1]
        u, v = image_pts[i][0], image_pts[i][1]
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    
    A = np.array(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1,:] / Vh[-1,-1]
    H = L.reshape(3, 3)
    
    return H

image1_pts = np.array([[100, 100], [200, 100], [200, 200], [100, 200]])  # Example points in image1
image2_pts = np.array([[150, 150], [250, 150], [250, 250], [150, 250]])  # Corresponding points in image2

H=computeH(image1_pts,image2_pts)

#translate homography
#p’=Hp, where H is a 3x3 matrix with 8 degrees of freedom

def translate_homography(H, tx, ty):
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    translated_H = np.dot(translation_matrix, H)
    return translated_H

translated_H = translate_homography(H, 50, 50)


#A.3: Warp the Images
#imwarped_nn = warpImageNearestNeighbor(im,H)
#imwarped_bil = warpImageBilinear(im,H)

def warpImageNearestNeighbor(im, H):
    # Get the dimensions of the input image
    h, w = im.shape[:2]
    
    # Create an output image of the same size
    imwarped = np.zeros_like(im)
    
    # Inverse of the homography matrix
    H_inv = np.linalg.inv(H)
    
    # all output pixel coords as a 3xN homogeneous array
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    p = np.stack([xs.ravel(), ys.ravel(), np.ones(h*w)])
    p_prime = H_inv @ p
    p_prime /= p_prime[2]

    x_prime = np.round(p_prime[0]).astype(int)
    y_prime = np.round(p_prime[1]).astype(int)
    valid = (0 <= x_prime) & (x_prime < w) & (0 <= y_prime) & (y_prime < h)

    imwarped[ys.ravel()[valid], xs.ravel()[valid]] = im[y_prime[valid], x_prime[valid]]
    return imwarped

def warpImageBilinear(im, H):
    # Get the dimensions of the input image
    h, w = im.shape[:2]
    
    # Create an output image of the same size
    imwarped = np.zeros_like(im)
    
    # Inverse of the homography matrix
    H_inv = np.linalg.inv(H)
    
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    p = np.stack([xs.ravel(), ys.ravel(), np.ones(h*w)])
    p_prime = H_inv @ p
    p_prime /= p_prime[2]

    x_prime, y_prime = p_prime[0], p_prime[1]
    valid = (0 <= x_prime) & (x_prime < w-1) & (0 <= y_prime) & (y_prime < h-1)
    x_prime, y_prime = x_prime[valid], y_prime[valid]

    x0, y0 = np.floor(x_prime).astype(int), np.floor(y_prime).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    dx, dy = (x_prime - x0)[:, None], (y_prime - y0)[:, None]  # [:, None] broadcasts over color channels

    imwarped[ys.ravel()[valid], xs.ravel()[valid]] = (im[y0, x0] * (1 - dx) * (1 - dy) +
                                                      im[y0, x1] * dx * (1 - dy) +
                                                      im[y1, x0] * (1 - dx) * dy +
                                                      im[y1, x1] * dx * dy)
    return imwarped

im2warped_nn = warpImageNearestNeighbor(image2,H)

im2warped_bil = warpImageBilinear(image2,H)

 
#A.4: Blend the Images into a Mosaic
#find laplasian pyramid of image1 nd image2 warped
img1_laplasian = cv2.pyrDown(image1)
img2_laplasian = cv2.pyrDown(im2warped_bil)

#blend image1 and warped image2 into a mosaic
belnded_image = np.maximum(img1_laplasian, img2_laplasian)

#upscale"

#display
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.imshow(belnded_image, cmap="gray")
plt.title("Blended Image")
plt.colorbar()
plt.show()

#A.5: Bells & Whistles (Extra for CS180, Mandatory for CS280A)