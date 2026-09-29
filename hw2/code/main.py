###################################################################################################
# Part 1: Fun with Filters
###################################################################################################

##########################################
## Part 1.1: Convolutions from Scratch! ##
##########################################

import numpy as np 
from scipy.signal import convolve2d

# D_x = [1, 0, -1]
# D_y = [1, 0, -1] transpose

#1.1.1: Imprement for loops for convoltion
#4 fileters
def convolce2d_4loops(image, kernel, padding="same"):
    """
    need four for loops
    1. loop over output rows
    2. loop output columns
    3. loop over filter rows
    4. loop over filter columns
    image: 2D NumPy array
    kernel: 2D NumPy array
    padding: "same" or "full"
    """
    kh, kw = kernel.shape

    #flip kernel
    kernel = np.flip(kernel)
    if padding == "same":
        pad_h = kh // 2
        pad_w = kw // 2
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    elif padding == "full":
        pad_h = kh - 1
        pad_w = kw - 1
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    else:
        raise ValueError("Padding must be 'same' or 'full'")
    

    if padding == "same":
        output_h, output_w = image.shape
    elif padding == "full":
        output_h, output_w = (image.shape[0] + kh - 1, image.shape[1] + kw - 1)
    
    output = np.zeros((output_h, output_w))

    #four forloops
    for i in range(output_h):
        for j in range(output_w):
            for m in range(kh):
                for n in range(kw):
                    output[i, j] += padded_image[i + m, j + n] * kernel[m, n]
    
    return output

#2 filters
def convolce2d_2loops(image, kernel, padding="same"):
    """
    need four for loops
    3. loop over filter rows
    4. loop over filter columns
    --> can be done with np array operations
    image: 2D NumPy array
    kernel: 2D NumPy array
    padding: "same" or "full"
    """
    kh, kw = kernel.shape

    #flip kernel
    kernel = np.flip(kernel)
    if padding == "same":
        pad_h = kh // 2
        pad_w = kw // 2
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    elif padding == "full":
        pad_h = kh - 1
        pad_w = kw - 1
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    else:
        raise ValueError("Padding must be 'same' or 'full'")
    

    if padding == "same":
        output_h, output_w = image.shape
    elif padding == "full":
        output_h, output_w = (image.shape[0] + kh - 1, image.shape[1] + kw - 1)
    
    output = np.zeros((output_h, output_w))

    #two forloops
    for i in range(output_h):
        for j in range(output_w):
            window = padded_image[i:i+kh, j:j+kw]
            output[i, j] = np.sum(window * kernel)
    return output


# #1.1.2: testing on my image 
from skimage.io import imread
import matplotlib.pyplot as plt

image = imread("data/my_photo.jpg", as_gray=True)
# print(image.shape)

kernel = np.ones((3, 3)) / 9  

#scipy
scipy_result = convolve2d(image, kernel, mode="same", boundary="fill", fillvalue=0)

my_result = convolce2d_2loops(image, kernel, padding="same")

# print("Maximum difference:", np.max(np.abs(my_result - scipy_result)))

#check if my image is normalized and if not normalize
image = image.astype(float)
if image.max() > 1:
    image /= 255.0

import time

# use a crop so the 4-loop version finishes in reasonable time
test_image = image[:256, :256]
test_kernel = np.ones((9, 9)) / 81   # 9x9 box filter

start = time.perf_counter()
result_4 = convolce2d_4loops(test_image, test_kernel, padding="same")
time_4 = time.perf_counter() - start

start = time.perf_counter()
result_2 = convolce2d_2loops(test_image, test_kernel, padding="same")
time_2 = time.perf_counter() - start

start = time.perf_counter()
result_scipy = convolve2d(test_image, test_kernel, mode="same", boundary="fill", fillvalue=0)
time_scipy = time.perf_counter() - start

print(f"Image {test_image.shape}, kernel {test_kernel.shape}")
print(f"4 loops: {time_4:.3f} s")
print(f"2 loops: {time_2:.3f} s")
print(f"SciPy:   {time_scipy:.5f} s")
print("Max diff (4 loops vs SciPy):", np.max(np.abs(result_4 - result_scipy)))
print("Max diff (2 loops vs SciPy):", np.max(np.abs(result_2 - result_scipy)))

# #display
# plt.imshow(image, cmap="gray")
# plt.axis("off")

# #1.1.3: box filter (9x9)
box_filter = np.ones((9, 9)) / 81
box_result = convolce2d_2loops(image, box_filter, padding="same")

# #display
# plt.imshow(box_result, cmap="gray")
# plt.axis("off")

# #1.1.4: funite difference
# #D_x and D_y
D_x = np.array([[-1, 0, 1]])
D_y = np.array([[-1],[0],[1]])

Dx_result = convolce2d_2loops(
    image,
    D_x,
    padding="same"
)

Dy_result = convolce2d_2loops(
    image,
    D_y,
    padding="same"
)

# #display
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# axes[0].imshow(image, cmap="gray")
# axes[0].set_title("Original")

# axes[1].imshow(Dx_result, cmap="gray")
# axes[1].set_title("D_x")

# axes[2].imshow(Dy_result, cmap="gray")
# axes[2].set_title("D_y")

# for ax in axes:
#     ax.axis("off")

# plt.show()

#########################################
## Part 1.2 Finite Difference Operator ##
#########################################

cameraman = plt.imread("data/cameraman.png")

#1.2.1 Convert to grayscale if necessary
if cameraman.ndim == 3:
    cameraman = cameraman.mean(axis=2)

cameraman = cameraman.astype(float)

#1.2.2 Normalize if image is in [0, 255]
if cameraman.max() > 1:
    cameraman /= 255.0

# plt.figure(figsize=(6, 6))
# plt.imshow(cameraman, cmap="gray")
# plt.title("Original Cameraman")
# plt.axis("off")

# #1.2.3 partial derivatives

partial_x = convolve2d(
    cameraman,
    D_x,
    mode="same",
    boundary="fill",
    fillvalue=0
)

partial_y = convolve2d(
    cameraman,
    D_y,
    mode="same",
    boundary="fill",
    fillvalue=0
)

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# axes[0].imshow(partial_x, cmap="gray")
# axes[0].set_title("Partial Derivative in X")
# axes[0].axis("off")

# axes[1].imshow(partial_y, cmap="gray")
# axes[1].set_title("Partial Derivative in Y")
# axes[1].axis("off")

# plt.show()

# #1.2.4 gradient magnitude

gradient_magnitude = np.sqrt(partial_x**2 + partial_y**2)
gradient_display = gradient_magnitude / gradient_magnitude.max()
# plt.figure(figsize=(6, 6))
# plt.imshow(gradient_magnitude, cmap="gray")
# plt.title("Gradient Magnitude")
# plt.axis("off")

# #1.2.5 binalize

threshold = 0.2
edge_image = gradient_display > threshold

# plt.figure(figsize=(6, 6))
# plt.imshow(edge_image, cmap="gray")
# plt.title(f"Edges, Threshold = {threshold}")
# plt.axis("off")

##################################################
## Part 1.3 Derivative of Gaussian (DoG) Filter ##
##################################################

#1.3.1 create gaussian filter

import cv2

ksize = 9
sigma = 2
gaussian_1d = cv2.getGaussianKernel(ksize, sigma)
gaussian_2d = gaussian_1d @ gaussian_1d.T

# print(gaussian_2d.shape)
# print("Sum:", gaussian_2d.sum())

# plt.figure(figsize=(6, 6))
# plt.imshow(gaussian_2d, cmap="gray")
# plt.title("9×9 Gaussian Filter")
# plt.colorbar()
# plt.show()

#1.3.2 blur image
blurred = convolve2d(cameraman, gaussian_2d, mode="same", boundary="fill", fillvalue=0)

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# axes[0].imshow(cameraman, cmap="gray")
# axes[0].set_title("Original")

# axes[1].imshow(blurred, cmap="gray")
# axes[1].set_title("Gaussian Blurred")

# for ax in axes:
#     ax.axis("off")

# plt.show()

#1.3.3 Derivative of Gaussian
blurred_dx = convolve2d(
    blurred,
    D_x,
    mode="same",
    boundary="fill",
    fillvalue=0
)

blurred_dy = convolve2d(
    blurred,
    D_y,
    mode="same",
    boundary="fill",
    fillvalue=0
)

#1.3.4 Gradient Magnitude
blurred_gradient = np.sqrt(
    blurred_dx**2 + blurred_dy**2
)

blurred_gradient_display = (
    blurred_gradient / blurred_gradient.max()
)

# plt.figure(figsize=(6, 6))
# plt.imshow(blurred_gradient_display, cmap="gray")
# plt.title("Gradient Magnitude After Gaussian Blur")
# plt.axis("off")

# plt.show() 

#1.3.5 binalize
threshold = 0.2

blurred_edges = blurred_gradient_display > threshold

# plt.figure(figsize=(6, 6))
# plt.imshow(blurred_edges, cmap="gray")
# plt.title(f"DoG-style Edges, Threshold = {threshold}")
# plt.axis("off")

# plt.show() 

#1.3.6 derivative of gaussian (DoG)

DoG_x = convolve2d(
    gaussian_2d,
    D_x,
    mode="full"
)

DoG_y = convolve2d(
    gaussian_2d,
    D_y,
    mode="full"
)

print("Gaussian:", gaussian_2d.shape)
print("DoG_x:", DoG_x.shape)
print("DoG_y:", DoG_y.shape)

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# axes[0].imshow(DoG_x, cmap="gray")
# axes[0].set_title("Derivative of Gaussian - Dx")
# axes[0].axis("off")

# axes[1].imshow(DoG_y, cmap="gray")
# axes[1].set_title("Derivative of Gaussian - Dy")
# axes[1].axis("off")

# plt.show()

#1.3.7 verify that DoG is equivalent to blurring then taking derivative
dog_x_result = convolve2d(
    cameraman,
    DoG_x,
    mode="same",
    boundary="fill",
    fillvalue=0
)

dog_y_result = convolve2d(
    cameraman,
    DoG_y,
    mode="same",
    boundary="fill",
    fillvalue=0
)

dog_gradient = np.sqrt(dog_x_result**2 + dog_y_result**2)

dog_gradient_display = dog_gradient / dog_gradient.max()

difference = np.abs(dog_gradient_display - blurred_gradient_display)

print("Maximum difference:", difference.max())
print("Mean difference:", difference.mean())

# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# axes[0].imshow(blurred_gradient_display, cmap="gray")
# axes[0].set_title("Blur → Derivative")
# axes[0].axis("off")

# axes[1].imshow(dog_gradient_display, cmap="gray")
# axes[1].set_title("Direct DoG")
# axes[1].axis("off")

# axes[2].imshow(difference, cmap="gray")
# axes[2].set_title("Absolute Difference")
# axes[2].axis("off")

# plt.show()

#################################
## [Optional] Bells & Whistles ##
#################################

###################################################################################################
# Part 2: Fun with Frequencies! 
###################################################################################################

##################################
## Part 2.1: Image "Sharpening" ##
##################################

#2.1.1 load taj

taj = plt.imread("data/taj.png")

# Convert RGB to grayscale only if desired.
# For sharpening, keeping RGB is also possible, but we'll
# start with grayscale for consistency with earlier parts.
if taj.ndim == 3:
    taj_gray = np.mean(taj[..., :3], axis=2)
else:
    taj_gray = taj

taj_gray = taj_gray.astype(float)

if taj_gray.max() > 1:
    taj_gray /= 255.0

# plt.figure(figsize=(6, 6))
# plt.imshow(taj_gray, cmap="gray")
# plt.title("Original Taj")
# plt.axis("off")

#2.1.2 gaussian filter
ksize = 9
sigma = 2

gaussian_1d = cv2.getGaussianKernel(ksize, sigma)
gaussian = gaussian_1d @ gaussian_1d.T

print(gaussian.sum())

#2.1.3 unsharpening filter
alpha = 1.5

impulse = np.zeros_like(gaussian)
center = ksize // 2
impulse[center, center] = 1

unsharp_filter = (1 + alpha) * impulse - alpha * gaussian

print(unsharp_filter)

taj_sharp = convolve2d(
    taj_gray,
    unsharp_filter,
    mode="same",
    boundary="symm"
)
taj_sharp = np.clip(taj_sharp, 0, 1)

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# axes[0].imshow(taj_gray, cmap="gray")
# axes[0].set_title("Original")

# axes[1].imshow(taj_sharp, cmap="gray")
# axes[1].set_title(f"Sharpened, α = {alpha}")

# for ax in axes:
#     ax.axis("off")

# plt.show()

#2.1.4 sharpening a blurred image

original = taj_gray.copy()

#blur
blurred = convolve2d(
    original,
    gaussian,
    mode="same",
    boundary="symm"
)

#sharpen
alpha = 1.5

unsharp_filter = (1 + alpha) * impulse - alpha * gaussian

recovered = convolve2d(
    blurred,
    unsharp_filter,
    mode="same",
    boundary="symm"
)

recovered = np.clip(recovered, 0, 1)

# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# axes[0].imshow(original, cmap="gray")
# axes[0].set_title("Original Sharp")

# axes[1].imshow(blurred, cmap="gray")
# axes[1].set_title("Blurred")

# axes[2].imshow(recovered, cmap="gray")
# axes[2].set_title("Sharpened After Blur")

# for ax in axes:
#     ax.axis("off")

# plt.show()


#############################
## Part 2.2: Hybrid Images ##
#############################

#2.2.1vload images
from align_image_code import align_images

# high sf
#im1 = plt.imread('data/DerekPicture.jpg') / 255.
im1 = plt.imread('data/ironman.jpg') / 255.
# low sf
#im2 = plt.imread('data/nutmeg.jpg') / 255.
im2 = plt.imread('data/tonystark.webp') / 255.

#align images (this code is provided, but may be improved)
im1_aligned, im2_aligned = align_images(im1, im2)

## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies


# #2.2.1 helper
# #gaussian 2d filter helper
# def gaussian_filter_2d(ksize, sigma):
#     kernel_1d = cv2.getGaussianKernel(ksize, sigma)
#     return kernel_1d @ kernel_1d.T

# #gaussian blur helper
# def gaussian_blur(image, ksize, sigma):
#     kernel = gaussian_filter_2d(ksize, sigma)
    
#     return convolve2d(
#         image,
#         kernel,
#         mode="same",
#         boundary="symm"
#     )

# #gaussian blur for RGB helper
# def gaussian_blur_color(image, ksize, sigma):
#     kernel = gaussian_filter_2d(ksize, sigma)
    
#     result = np.zeros_like(image, dtype=float)
    
#     for channel in range(image.shape[2]):
#         result[..., channel] = convolve2d(
#             image[..., channel],
#             kernel,
#             mode="same",
#             boundary="symm"
#         )
    
#     return result

# #low pass/high pass
# sigma_low = 5
# sigma_high = 3

# low_pass = gaussian_blur_color(
#     im2_aligned,
#     ksize=21,
#     sigma=sigma_low
# )

# blurred_high = gaussian_blur_color(
#     im1_aligned,
#     ksize=21,
#     sigma=sigma_high
# )

# high_pass = im1_aligned - blurred_high

# hybrid = low_pass + high_pass
# hybrid = np.clip(hybrid, 0, 1)

# fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# axes[0].imshow(im2)
# axes[0].set_title("Low-Frequency Image")

# axes[1].imshow(im1)
# axes[1].set_title("High-Frequency Image")

# axes[2].imshow(hybrid)
# axes[2].set_title("Hybrid Image")

# for ax in axes:
#     ax.axis("off")

# plt.show()

#2.2.2 fourier analysis
# def fourier_magnitude(image):
#     if image.ndim == 3:
#         image = np.mean(image, axis=2)
        
#     fft = np.fft.fft2(image)
#     fft_shifted = np.fft.fftshift(fft)
    
#     return np.log1p(np.abs(fft_shifted))

# images = [
#     im2,
#     im1,
#     low_pass,
#     high_pass,
#     hybrid
# ]

# titles = [
#     "Input 1",
#     "Input 2",
#     "Low-Pass Image",
#     "High-Pass Image",
#     "Hybrid Image"
# ]

# fig, axes = plt.subplots(1, 5, figsize=(20, 4))

# for ax, image, title in zip(axes, images, titles):
#     magnitude = fourier_magnitude(image)
    
#     ax.imshow(magnitude, cmap="gray")
#     ax.set_title(title)
#     ax.axis("off")

# plt.show()

#2.2.3 frequency analysis
# fig, axes = plt.subplots(1, 5, figsize=(20, 4))

# for ax, image, title in zip(axes, images, titles):
#     magnitude = fourier_magnitude(image)
    
#     ax.imshow(magnitude, cmap="gray")
#     ax.set_title(f"FFT: {title}")
#     ax.axis("off")

# plt.tight_layout()
# plt.show()

#################################
## [Optional] Bells & Whistles ##
#################################

# Multi-resolution blending and the oraple journey
#############################################
## Part 2.3: Gaussian and leplavoam stacks ##
#############################################

#2.3.1 load images

apple_image = plt.imread("data/apple.png").astype(float)
orange_image = plt.imread("data/orange.png").astype(float)
#only keep rgb (drop the alpha channel)
apple_image = apple_image[..., :3].astype(float)
orange_image = orange_image[..., :3].astype(float)
#normalize
if apple_image.max() > 1:
    apple_image /= 255.0

if orange_image.max() > 1:
    orange_image /= 255.0
#crop to a common size
h = min(apple_image.shape[0], orange_image.shape[0])
w = min(apple_image.shape[1], orange_image.shape[1])
apple_image = apple_image[:h, :w]
orange_image = orange_image[:h, :w]


#2.3.2 Gaussian stack

def make_gaussian_kernel(ksize=9, sigma=2):
    kernel_1d = cv2.getGaussianKernel(ksize, sigma)
    return kernel_1d @ kernel_1d.T
#greyscale
def gaussian_blur_gray(image, kernel):
    return convolve2d(
        image,
        kernel,
        mode="same",
        boundary="symm"
    )
#RGB (named differently from the 2.2 gaussian_blur_color, which takes ksize/sigma)
def gaussian_blur_rgb(image, kernel):
    result = np.zeros_like(image, dtype=float)

    for c in range(image.shape[2]):
        result[:, :, c] = convolve2d(
            image[:, :, c],
            kernel,
            mode="same",
            boundary="symm"
        )

    return result

#implement gaussian stack
def gaussian_stack(image, num_levels=5, ksize=9, sigma=2):
    kernel = make_gaussian_kernel(ksize, sigma)

    stack = [image.astype(float)]

    for i in range(1, num_levels):
        previous = stack[-1]

        if image.ndim == 2:
            blurred = gaussian_blur_gray(previous, kernel)
        else:
            blurred = gaussian_blur_rgb(previous, kernel)

        stack.append(blurred)

    return np.array(stack)

G_apple = gaussian_stack(apple_image, num_levels=5, ksize=9, sigma=2)
G_orange = gaussian_stack(orange_image, num_levels=5, ksize=9, sigma=2)

print(G_apple.shape)
print(G_orange.shape)

# fig, axes = plt.subplots(1, len(G_apple), figsize=(20, 5))

# for i, ax in enumerate(axes):
#     if G_apple[i].ndim == 2:
#         ax.imshow(G_apple[i], cmap="gray")
#     else:
#         ax.imshow(np.clip(G_apple[i], 0, 1))

#     ax.set_title(f"Gaussian Level {i}")
#     ax.axis("off")

# plt.show()

# fig, axes = plt.subplots(1, len(G_apple), figsize=(20, 5))

# for i, ax in enumerate(axes):
#     if G_orange[i].ndim == 2:
#         ax.imshow(G_orange[i], cmap="gray")
#     else:
#         ax.imshow(np.clip(G_orange[i], 0, 1))

#     ax.set_title(f"Gaussian Level {i}")
#     ax.axis("off")

# plt.show()

#2.3.3 laplasian stack
def laplacian_stack(gaussian_stack):
    num_levels = len(gaussian_stack)

    laplacian = []

    for i in range(num_levels - 1):
        laplacian.append(
            gaussian_stack[i] - gaussian_stack[i + 1]
        )

    # Last level contains the lowest-frequency information
    laplacian.append(gaussian_stack[-1])

    return np.array(laplacian)

L_apple = laplacian_stack(G_apple)
L_orange = laplacian_stack(G_orange)

print(L_apple.shape)
print(L_orange.shape)

# fig, axes = plt.subplots(1, len(L_apple), figsize=(20, 5))

# for i, ax in enumerate(axes):
#     if L_apple[i].ndim == 2:
#         ax.imshow(L_apple[i], cmap="gray")
#     else:
#         ax.imshow(np.clip(L_apple[i] + 0.5, 0, 1))

#     ax.set_title(f"Laplacian Level {i}")
#     ax.axis("off")

# fig, axes = plt.subplots(1, len(L_orange), figsize=(20, 5))

# for i, ax in enumerate(axes):
#     if L_orange[i].ndim == 2:
#         ax.imshow(L_orange[i], cmap="gray")
#     else:
#         ax.imshow(np.clip(L_orange[i] + 0.5, 0, 1))

#     ax.set_title(f"Laplacian Level {i}")
#     ax.axis("off")

# plt.show()

# fig, axes = plt.subplots(1, len(L_apple), figsize=(20, 5))

# for i, ax in enumerate(axes):
#     level = L_apple[i]

#     if level.ndim == 3:
#         minimum = level.min()
#         maximum = level.max()

#         display = (level - minimum) / (maximum - minimum + 1e-8)
#         ax.imshow(display)
#     else:
#         ax.imshow(level, cmap="gray")

#     ax.set_title(f"Laplacian Level {i}")
#     ax.axis("off")

# plt.show()

# fig, axes = plt.subplots(1, len(L_orange), figsize=(20, 5))

# for i, ax in enumerate(axes):
#     level = L_orange[i]

#     if level.ndim == 3:
#         minimum = level.min()
#         maximum = level.max()

#         display = (level - minimum) / (maximum - minimum + 1e-8)
#         ax.imshow(display)
#     else:
#         ax.imshow(level, cmap="gray")

#     ax.set_title(f"Laplacian Level {i}")
#     ax.axis("off")

# plt.show()

#2.3.4 reconstruct image from laplacian stack
reconstructed_apple = np.sum(L_apple, axis=0)
reconstructed_orange = np.sum(L_orange, axis=0)

# print("Apple max reconstruction error:",
#       np.max(np.abs(reconstructed_apple - apple_image)))
# print("Orange max reconstruction error:",
#       np.max(np.abs(reconstructed_orange - orange_image)))

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# axes[0].imshow(np.clip(reconstructed_apple, 0, 1))
# axes[0].set_title("Reconstructed Apple")

# axes[1].imshow(np.clip(reconstructed_orange, 0, 1))
# axes[1].set_title("Reconstructed Orange")

# for ax in axes:
#     ax.axis("off")

# plt.show()

#############################################################
## Part 2.4: Multiresolution Blending (a.k.a. the oraple!) ##
#############################################################

#images
#apple_image
#orange_image

#vertical mask
H, W = apple_image.shape[:2]
mask = np.zeros((H, W))
mask[:, :W // 2] = 1

# plt.figure(figsize=(8, 5))
# plt.imshow(mask, cmap="gray")
# plt.title("Vertical Step Mask")
# plt.axis("off")

G_mask = gaussian_stack(mask, num_levels=5, ksize=31, sigma=5)

#blend
L_blended = []

for i in range(5):
    m = G_mask[i]

    # Add channel dimension for RGB images
    if L_apple[i].ndim == 3:
        m = m[:, :, np.newaxis]

    blended_level = (
        m * L_apple[i]
        + (1 - m) * L_orange[i]
    )

    L_blended.append(blended_level)

L_blended = np.array(L_blended)

#create oraple
oraple = np.sum(L_blended, axis=0)
oraple = np.clip(oraple, 0, 1)

# plt.figure(figsize=(8, 6))
# plt.imshow(oraple)
# plt.title("Oraple — Multiresolution Blend")
# plt.axis("off")

#neive blending
naive = (mask[:, :, np.newaxis] * apple_image + (1 - mask[:, :, np.newaxis]) * orange_image)

# fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# axes[0].imshow(naive)
# axes[0].set_title("Naive Cut-and-Paste")

# axes[1].imshow(oraple)
# axes[1].set_title("Multiresolution Blend")

# for ax in axes:
#     ax.axis("off")

# plt.show()

#gaussian mask stack

# fig, axes = plt.subplots(1, 5, figsize=(20, 4))

# for i, ax in enumerate(axes):
#     ax.imshow(G_mask[i], cmap="gray")
#     ax.set_title(f"Mask Level {i}")
#     ax.axis("off")

# plt.show()

#irregular mask

Y, X = np.ogrid[:H, :W]

center_x = W // 2
center_y = H // 2
radius = min(H, W) // 3

mask_circle = (
    (X - center_x)**2 +
    (Y - center_y)**2
    < radius**2
)

mask_circle = mask_circle.astype(float)

# plt.figure(figsize=(6, 6))
# plt.imshow(mask_circle, cmap="gray")
# plt.title("Circular Mask")
# plt.axis("off")