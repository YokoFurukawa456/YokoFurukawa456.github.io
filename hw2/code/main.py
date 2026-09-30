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
# plt.savefig("save/1.2_gradient_magnitude.png", bbox_inches="tight")

# #1.2.5 binalize

threshold = 0.2
edge_image = gradient_display > threshold

# plt.figure(figsize=(6, 6))
# plt.imshow(edge_image, cmap="gray")
# plt.title(f"Edges, Threshold = {threshold}")
# plt.axis("off")
# plt.savefig("save/1.2_gradient_magnitude_binarized.png", bbox_inches="tight")

# plt.show()

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
blurred = convolve2d(cameraman, gaussian_2d, mode="same", boundary="symm")

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
    boundary="symm"
)

blurred_dy = convolve2d(
    blurred,
    D_y,
    mode="same",
    boundary="symm"
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

# print("Gaussian:", gaussian_2d.shape)
# print("DoG_x:", DoG_x.shape)
# print("DoG_y:", DoG_y.shape)

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
    boundary="symm"
)

dog_y_result = convolve2d(
    cameraman,
    DoG_y,
    mode="same",
    boundary="symm"
)

dog_gradient = np.sqrt(dog_x_result**2 + dog_y_result**2)

dog_gradient_display = dog_gradient / dog_gradient.max()

difference = np.abs(dog_gradient_display - blurred_gradient_display)

# print("Maximum difference:", difference.max())
# print("Mean difference:", difference.mean())

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

# plt.savefig("save/1.3_compare.png", bbox_inches="tight")

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

# taj = plt.imread("data/taj.png")
taj = plt.imread("data/sf.JPG")
# phone photo is 4032x3024; shrink it so a sigma=2 blur is actually visible (and runs faster)
taj = cv2.resize(taj, (taj.shape[1] // 4, taj.shape[0] // 4), interpolation=cv2.INTER_AREA)

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

# print(gaussian.sum())

#2.1.3 unsharpening filter
alpha = 1.5

impulse = np.zeros_like(gaussian)
center = ksize // 2
impulse[center, center] = 1

unsharp_filter = (1 + alpha) * impulse - alpha * gaussian

# print(unsharp_filter)

taj_sharp = convolve2d(
    taj_gray,
    unsharp_filter,
    mode="same",
    boundary="symm"
)
taj_sharp = np.clip(taj_sharp, 0, 1)

#2.1.3b color version: save original / blurred / high frequency / sharpened
taj_rgb = taj[..., :3].astype(float)
# JPGs load as 0-255, PNGs as 0-1
if taj_rgb.max() > 1:
    taj_rgb /= 255.0

# apply a 2D filter to each color channel separately
def filter_rgb(image, kernel):
    return np.stack([
        convolve2d(image[..., c], kernel, mode="same", boundary="symm")
        for c in range(3)
    ], axis=2)

taj_blurred = filter_rgb(taj_rgb, gaussian)
taj_high = taj_rgb - taj_blurred
taj_sharp_rgb = np.clip(filter_rgb(taj_rgb, unsharp_filter), 0, 1)

# fig, axes = plt.subplots(1, 4, figsize=(24, 6))

# axes[0].imshow(taj_rgb)
# axes[0].set_title("Original")

# axes[1].imshow(np.clip(taj_blurred, 0, 1))
# axes[1].set_title(f"Blurred (σ = {sigma})")

# # high frequencies are centered around 0, so shift by 0.5 to make them visible
# axes[2].imshow(np.clip(taj_high + 0.5, 0, 1))
# axes[2].set_title("High Frequency (+0.5)")

# axes[3].imshow(taj_sharp_rgb)
# axes[3].set_title(f"Sharpened (α = {alpha})")

# for ax in axes:
#     ax.axis("off")

#plt.savefig("save/2.1_taj.png", bbox_inches="tight")
# plt.savefig("save/2.1_sf.png", bbox_inches="tight")
# plt.show()

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

#2.1.5 color Taj: vary the sharpening amount
taj_color = plt.imread("data/taj.png")[..., :3].astype(float)

alphas = [0.5, 1, 2, 5]

# fig, axes = plt.subplots(1, len(alphas) + 1, figsize=(25, 6))

# axes[0].imshow(taj_color)
# axes[0].set_title("Original")

# for ax, a in zip(axes[1:], alphas):
#     filter_a = (1 + a) * impulse - a * gaussian
#     ax.imshow(np.clip(filter_rgb(taj_color, filter_a), 0, 1))
#     ax.set_title(f"α = {a}")

# for ax in axes:
#     ax.axis("off")

# plt.savefig("save/2.1_taj_alpha.png", bbox_inches="tight")
# plt.show()

#2.1.6 color Taj: blur a sharp image, then try to re-sharpen it
taj_color_blurred = filter_rgb(taj_color, gaussian)
taj_color_recovered = np.clip(filter_rgb(taj_color_blurred, unsharp_filter), 0, 1)

# fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# axes[0].imshow(taj_color)
# axes[0].set_title("Original Sharp")

# axes[1].imshow(np.clip(taj_color_blurred, 0, 1))
# axes[1].set_title(f"Blurred (σ = {sigma})")

# axes[2].imshow(taj_color_recovered)
# axes[2].set_title(f"Re-sharpened (α = {alpha})")

# for ax in axes:
#     ax.axis("off")

# plt.savefig("save/2.1_taj_resharpen.png", bbox_inches="tight")
# plt.show()


#############################
## Part 2.2: Hybrid Images ##
#############################

#2.2.1vload images
from align_image_code import align_images

# high sf
#im1 = plt.imread('data/derekPicture.jpg') / 255.
#im1 = plt.imread('data/ironman.jpg') / 255.
im1 = plt.imread('data/hurk.png')[..., :3]  # PNG: already 0-1, drop alpha
# low sf
#im2 = plt.imread('data/nutmeg.jpg') / 255.
#im2 = plt.imread('data/tonystark.webp') / 255.
im2 = plt.imread('data/bruce.png')[..., :3]

#align images (this code is provided, but may be improved)
im1_aligned, im2_aligned = align_images(im1, im2)

## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies


#2.2.1 helper
#gaussian 2d filter helper
def gaussian_filter_2d(ksize, sigma):
    kernel_1d = cv2.getGaussianKernel(ksize, sigma)
    return kernel_1d @ kernel_1d.T

#gaussian blur helper
def gaussian_blur(image, ksize, sigma):
    kernel = gaussian_filter_2d(ksize, sigma)
    
    return convolve2d(
        image,
        kernel,
        mode="same",
        boundary="symm"
    )

#gaussian blur for RGB helper
def gaussian_blur_color(image, ksize, sigma):
    kernel = gaussian_filter_2d(ksize, sigma)
    
    result = np.zeros_like(image, dtype=float)
    
    for channel in range(image.shape[2]):
        result[..., channel] = convolve2d(
            image[..., channel],
            kernel,
            mode="same",
            boundary="symm"
        )
    
    return result

#low pass/high pass
sigma_low = 5
sigma_high = 3

low_pass = gaussian_blur_color(
    im2_aligned,
    ksize=21,
    sigma=sigma_low
)

blurred_high = gaussian_blur_color(
    im1_aligned,
    ksize=21,
    sigma=sigma_high
)

high_pass = im1_aligned - blurred_high

hybrid = low_pass + high_pass
hybrid = np.clip(hybrid, 0, 1)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(im2)
axes[0].set_title("Low-Frequency Image")

axes[1].imshow(im1)
axes[1].set_title("High-Frequency Image")

axes[2].imshow(hybrid)
axes[2].set_title("Hybrid Image")

for ax in axes:
    ax.axis("off")

plt.show()

#2.2.4 save aligned images and final hybrids for each pair

# blur an RGB image with a ksize x ksize Gaussian
def hybrid_blur(image, ksize, sigma):
    kernel_1d = cv2.getGaussianKernel(ksize, sigma)
    kernel = kernel_1d @ kernel_1d.T
    return np.stack([
        convolve2d(image[..., c], kernel, mode="same", boundary="symm")
        for c in range(image.shape[2])
    ], axis=2)

def make_hybrid_image(im_low, im_high, ksize=21, sigma_low=5, sigma_high=3):
    low = hybrid_blur(im_low, ksize, sigma_low)
    high = im_high - hybrid_blur(im_high, ksize, sigma_high)
    return np.clip(low + high, 0, 1)

# alignment pads with black; keep only rows/columns where both images have content
def crop_padding(a, b):
    keep_rows = (a.sum(axis=(1, 2)) > 0) & (b.sum(axis=(1, 2)) > 0)
    keep_cols = (a.sum(axis=(0, 2)) > 0) & (b.sum(axis=(0, 2)) > 0)
    return a[keep_rows][:, keep_cols], b[keep_rows][:, keep_cols]

def load_rgb(path):
    im = plt.imread(path)[..., :3].astype(float)
    if im.max() > 1:
        im /= 255.0
    return im

# (name, low-frequency image, high-frequency image, saved alignment points)
# click the HIGH-frequency image first, then the LOW one (eyes, same order);
# after one run, paste the printed "Alignment points" in place of None
hybrid_pairs = [
    ("stark_ironman", "data/tonystark.webp", "data/ironman.jpg", None),
    ("derek_nutmeg", "data/nutmeg.jpg", "data/DerekPicture.jpg", None),
]

for name, low_path, high_path, pts in hybrid_pairs:
    im_low = load_rgb(low_path)
    im_high = load_rgb(high_path)

    high_aligned, low_aligned = align_images(im_high, im_low, pts)
    high_aligned, low_aligned = crop_padding(high_aligned, low_aligned)

    hybrid_result = make_hybrid_image(low_aligned, high_aligned)

    # aligned inputs, before filtering
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    axes[0].imshow(np.clip(low_aligned, 0, 1))
    axes[0].set_title("Aligned Low-Frequency Image")
    axes[1].imshow(np.clip(high_aligned, 0, 1))
    axes[1].set_title("Aligned High-Frequency Image")
    for ax in axes:
        ax.axis("off")
    plt.savefig(f"save/2.2_{name}_aligned.png", bbox_inches="tight")

    # originals and final hybrid
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    axes[0].imshow(im_low)
    axes[0].set_title("Low-Frequency Original")
    axes[1].imshow(im_high)
    axes[1].set_title("High-Frequency Original")
    axes[2].imshow(hybrid_result)
    axes[2].set_title("Hybrid")
    for ax in axes:
        ax.axis("off")
    plt.savefig(f"save/2.2_{name}_hybrid.png", bbox_inches="tight")

    # hybrid on its own, full resolution
    plt.imsave(f"save/2.2_{name}_hybrid_only.png", hybrid_result)

plt.show()

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

# print(G_apple.shape)
# print(G_orange.shape)

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

# print(L_apple.shape)
# print(L_orange.shape)

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

#2.4.1 recreate Figure 3.42: masked Laplacian levels of apple, orange, and their blend
masked_apple = [G_mask[i][:, :, np.newaxis] * L_apple[i] for i in range(5)]
masked_orange = [(1 - G_mask[i][:, :, np.newaxis]) * L_orange[i] for i in range(5)]

# Laplacian bands are centered at 0: map 0 to gray and use the same scale across a row
def show_band(band, scale):
    return np.clip(0.5 + band / (2 * scale), 0, 1)

fig, axes = plt.subplots(4, 3, figsize=(15, 16))

for r, level in enumerate([0, 2, 4]):
    trio = [masked_apple[level], masked_orange[level], L_blended[level]]

    if level == len(L_blended) - 1:
        # last level is the low-pass residual, which is a normal image
        images = [np.clip(t, 0, 1) for t in trio]
    else:
        scale = np.percentile(np.abs(np.concatenate([t.ravel() for t in trio])), 99.5)
        images = [show_band(t, scale) for t in trio]

    for c, title in enumerate(["Masked Apple", "Masked Orange", "Blended"]):
        axes[r, c].imshow(images[c])
        axes[r, c].set_title(f"Level {level}: {title}")

# bottom row: sum over all levels
# axes[3, 0].imshow(np.clip(np.sum(masked_apple, axis=0), 0, 1))
# axes[3, 0].set_title("Sum: Masked Apple")
# axes[3, 1].imshow(np.clip(np.sum(masked_orange, axis=0), 0, 1))
# axes[3, 1].set_title("Sum: Masked Orange")
# axes[3, 2].imshow(oraple)
# axes[3, 2].set_title("Sum: Oraple")

for ax in axes.ravel():
    ax.axis("off")

# plt.tight_layout()
# plt.savefig("save/2.4_figure342.png", bbox_inches="tight")
# plt.show()

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

plt.figure(figsize=(6, 6))
plt.imshow(mask_circle, cmap="gray")
plt.title("Circular Mask")
plt.axis("off")

#2.4.2 irregular mask: the Berkeley campus seen through Dr. Strange's portal
strange = plt.imread("data/drstrance.png")[..., :3].astype(float)
berkeley = plt.imread("data/berkeley.png")[..., :3].astype(float)
PH, PW = strange.shape[:2]

# shrink Berkeley and shift it so the Campanile sits inside the portal
scale = 380 / berkeley.shape[0]
berk_small = cv2.resize(berkeley, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
shift_x, shift_y = 34, 12   # pixels cut off the left / top
berk_crop = berk_small[shift_y:shift_y + PH, shift_x:shift_x + PW]
# fill the rest of the canvas by repeating edge pixels (it's outside the portal, so hidden)
berkeley_canvas = np.pad(
    berk_crop,
    ((0, PH - berk_crop.shape[0]), (0, PW - berk_crop.shape[1]), (0, 0)),
    mode="edge"
)

# same idea as the circular mask, but an ellipse (separate x / y radii) to match the portal
Yp, Xp = np.ogrid[:PH, :PW]
portal_cx, portal_cy = 175, 178
portal_rx, portal_ry = 125, 160

mask_portal = (
    ((Xp - portal_cx) / portal_rx)**2 +
    ((Yp - portal_cy) / portal_ry)**2
    < 1
).astype(float)

# Gaussian / Laplacian stacks for both images and the mask
G_berk = gaussian_stack(berkeley_canvas, num_levels=5, ksize=9, sigma=2)
G_strange = gaussian_stack(strange, num_levels=5, ksize=9, sigma=2)
L_berk = laplacian_stack(G_berk)
L_strange = laplacian_stack(G_strange)
G_portal = gaussian_stack(mask_portal, num_levels=5, ksize=31, sigma=5)

# blend each level: Berkeley inside the portal, Dr. Strange outside
masked_berk = [G_portal[i][:, :, np.newaxis] * L_berk[i] for i in range(5)]
masked_strange = [(1 - G_portal[i][:, :, np.newaxis]) * L_strange[i] for i in range(5)]
L_portal = [masked_berk[i] + masked_strange[i] for i in range(5)]

portal_blend = np.clip(np.sum(L_portal, axis=0), 0, 1)

# inputs, mask, and result
fig, axes = plt.subplots(2, 2, figsize=(16, 8))

axes[0, 0].imshow(strange)
axes[0, 0].set_title("Dr. Strange")
axes[0, 1].imshow(np.clip(berkeley_canvas, 0, 1))
axes[0, 1].set_title("Berkeley (resized and shifted)")
axes[1, 0].imshow(mask_portal, cmap="gray")
axes[1, 0].set_title("Elliptical Portal Mask")
axes[1, 1].imshow(portal_blend)
axes[1, 1].set_title("Multiresolution Blend")

for ax in axes.ravel():
    ax.axis("off")

plt.tight_layout()
plt.savefig("save/2.4_portal.png", bbox_inches="tight")
plt.imsave("save/2.4_portal_only.png", portal_blend)

# Laplacian stack of the blend, like Figure 10 in the paper
fig, axes = plt.subplots(4, 3, figsize=(18, 12))

for r, level in enumerate([0, 2, 4]):
    trio = [masked_berk[level], masked_strange[level], L_portal[level]]

    if level == 4:
        # last level is the low-pass residual, which is a normal image
        images = [np.clip(t, 0, 1) for t in trio]
    else:
        scale_band = np.percentile(np.abs(np.concatenate([t.ravel() for t in trio])), 99.5)
        images = [show_band(t, scale_band) for t in trio]

    for c, title in enumerate(["Masked Berkeley", "Masked Dr. Strange", "Blended"]):
        axes[r, c].imshow(images[c])
        axes[r, c].set_title(f"Level {level}: {title}")

axes[3, 0].imshow(np.clip(np.sum(masked_berk, axis=0), 0, 1))
axes[3, 0].set_title("Sum: Masked Berkeley")
axes[3, 1].imshow(np.clip(np.sum(masked_strange, axis=0), 0, 1))
axes[3, 1].set_title("Sum: Masked Dr. Strange")
axes[3, 2].imshow(portal_blend)
axes[3, 2].set_title("Sum: Blend")

for ax in axes.ravel():
    ax.axis("off")

plt.tight_layout()
plt.savefig("save/2.4_portal_stack.png", bbox_inches="tight")
plt.show()


#2.4.3 irregular mask x6: Thanos's face inside each Infinity Stone
hand = plt.imread("data/thanoshand.png")[..., :3].astype(float)
thanos = plt.imread("data/thanos.png")[..., :3].astype(float)

# upscale the gauntlet 2x so the tiny faces keep some detail
up = 2
hand = cv2.resize(hand, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
hand = np.clip(hand, 0, 1)
HH, HW = hand.shape[:2]

# crop just the face out of the Thanos photo
thanos_face = thanos[20:300, 225:420]

# (name, center x, center y, x radius, y radius) in original gauntlet pixels
stones = [
    ("soul",    23, 291, 10, 17),
    ("reality", 47, 263, 13, 24),
    ("space",   94, 243, 15, 27),
    ("power",  151, 222, 21, 25),
    ("mind",    92, 337, 24, 31),
    ("time",   278, 333, 21, 29),
]

# put the face, resized to cover one stone, onto a canvas the size of the gauntlet
def face_on_canvas(face, cx, cy, rx, ry):
    fh, fw = int(2.2 * ry), int(2.2 * rx)
    small = cv2.resize(face, (fw, fh), interpolation=cv2.INTER_AREA)
    top, left = cy - fh // 2, cx - fw // 2
    # fill the rest by repeating edge pixels (hidden by the mask anyway)
    return np.pad(
        small,
        ((top, HH - top - fh), (left, HW - left - fw), (0, 0)),
        mode="edge"
    )

# Bells & Whistles: tint each face with its stone's color (False = plain faces)
tint_faces = True
suffix = "" if tint_faces else "_notint"

Ys, Xs = np.ogrid[:HH, :HW]
stones_result = hand.copy()
stone_masks = np.zeros((HH, HW))

# same blend as the portal, repeated once per stone
for name, cx, cy, rx, ry in stones:
    cx, cy, rx, ry = cx * up, cy * up, rx * up, ry * up

    mask_stone = (
        ((Xs - cx) / rx)**2 +
        ((Ys - cy) / ry)**2
        < 1
    ).astype(float)
    stone_masks = np.maximum(stone_masks, mask_stone)

    face_canvas = face_on_canvas(thanos_face, cx, cy, rx, ry)

    # tint the face with the stone's color: keep the face's brightness detail,
    # and scale it so its average color inside the stone matches the original stone
    if tint_faces:
        inside = mask_stone > 0
        stone_color = hand[inside].mean(axis=0)
        face_gray = face_canvas.mean(axis=2)
        face_canvas = np.clip(
            face_gray[:, :, np.newaxis] / face_gray[inside].mean() * stone_color,
            0, 1
        )

    L_face = laplacian_stack(gaussian_stack(face_canvas, num_levels=5, ksize=9, sigma=2))
    L_base = laplacian_stack(gaussian_stack(stones_result, num_levels=5, ksize=9, sigma=2))
    # small stones -> smaller mask blur than the portal
    G_stone = gaussian_stack(mask_stone, num_levels=5, ksize=9, sigma=2)

    L_stone = [
        G_stone[i][:, :, np.newaxis] * L_face[i]
        + (1 - G_stone[i][:, :, np.newaxis]) * L_base[i]
        for i in range(5)
    ]
    stones_result = np.clip(np.sum(L_stone, axis=0), 0, 1)

fig, axes = plt.subplots(1, 4, figsize=(20, 9))

axes[0].imshow(thanos)
axes[0].set_title("Thanos")
axes[1].imshow(hand)
axes[1].set_title("Infinity Gauntlet")
axes[2].imshow(stone_masks, cmap="gray")
axes[2].set_title("Six Elliptical Stone Masks")
axes[3].imshow(stones_result)
axes[3].set_title("Multiresolution Blend")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig(f"save/2.4_thanos_stones{suffix}.png", bbox_inches="tight")
plt.imsave(f"save/2.4_thanos_stones{suffix}_only.png", stones_result)
plt.show()


#2.4.4 straight-line mask: half Peter Parker, half Spider-Man
peter = load_rgb("data/peter.png")
spider = load_rgb("data/spider.png")

# line up the eyes (pupils for Peter, eye lenses for Spider-Man), measured once by hand, as (x, y)
peter_eyes = ((190, 246), (290, 243))
spider_eyes = ((205, 145), (282, 145))
# x position of each nose's center line (between Peter's nostrils / Spider-Man's center web strand)
peter_nose_x, spider_nose_x = 238, 245

# align_images puts the midpoint of the two points at the image center, so slide each
# pair of eye points sideways until their midpoint is on the nose (spacing and angle stay the same)
def center_on_nose(eyes, nose_x):
    (x1, y1), (x2, y2) = eyes
    d = nose_x - (x1 + x2) / 2
    return (x1 + d, y1), (x2 + d, y2)

face_pts = center_on_nose(peter_eyes, peter_nose_x) + center_on_nose(spider_eyes, spider_nose_x)
peter_aligned, spider_aligned = align_images(peter, spider, face_pts)

# remove the black padding, cutting the SAME amount from left and right so the nose
# stays exactly in the middle column (top/bottom don't matter for a vertical seam)
def crop_padding_centered(a, b):
    keep_rows = (a.sum(axis=(1, 2)) > 0) & (b.sum(axis=(1, 2)) > 0)
    keep_cols = (a.sum(axis=(0, 2)) > 0) & (b.sum(axis=(0, 2)) > 0)

    def symmetric(keep):
        idx = np.where(keep)[0]
        c = max(idx[0], len(keep) - 1 - idx[-1])
        return slice(c, len(keep) - c)

    cols = symmetric(keep_cols)
    return a[keep_rows][:, cols], b[keep_rows][:, cols]

peter_aligned, spider_aligned = crop_padding_centered(peter_aligned, spider_aligned)
# the small rotation leaves thin black slivers at the edges; trim a margin (same on every side)
m = 16
peter_aligned, spider_aligned = peter_aligned[m:-m, m:-m], spider_aligned[m:-m, m:-m]

# both noses are now on the center column, so a vertical step mask at W // 2
# puts the seam right on the nose
FH, FW = peter_aligned.shape[:2]
mask_face = np.zeros((FH, FW))
mask_face[:, :FW // 2] = 1

G_peter = gaussian_stack(peter_aligned, num_levels=5, ksize=9, sigma=2)
G_spider = gaussian_stack(spider_aligned, num_levels=5, ksize=9, sigma=2)
L_peter = laplacian_stack(G_peter)
L_spider = laplacian_stack(G_spider)
G_face = gaussian_stack(mask_face, num_levels=5, ksize=31, sigma=5)

L_face_blend = [
    G_face[i][:, :, np.newaxis] * L_peter[i]
    + (1 - G_face[i][:, :, np.newaxis]) * L_spider[i]
    for i in range(5)
]
face_blend = np.clip(np.sum(L_face_blend, axis=0), 0, 1)

face_naive = mask_face[:, :, np.newaxis] * peter_aligned + (1 - mask_face[:, :, np.newaxis]) * spider_aligned

fig, axes = plt.subplots(1, 5, figsize=(25, 7))

axes[0].imshow(np.clip(peter_aligned, 0, 1))
axes[0].set_title("Peter (aligned)")
axes[1].imshow(np.clip(spider_aligned, 0, 1))
axes[1].set_title("Spider-Man (aligned)")
axes[2].imshow(mask_face, cmap="gray", vmin=0, vmax=1)
axes[2].set_title("Vertical Step Mask")
axes[3].imshow(np.clip(face_naive, 0, 1))
axes[3].set_title("Naive Cut-and-Paste")
axes[4].imshow(face_blend)
axes[4].set_title("Multiresolution Blend")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig("save/2.4_peter_spider.png", bbox_inches="tight")
plt.imsave("save/2.4_peter_spider_only.png", face_blend)
plt.show()
