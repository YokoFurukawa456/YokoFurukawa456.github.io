# Part 1: Fun with Filters

## Part 1.1: Convolutions from Scratch!
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

#scipy
scipy_result = convolve2d(image, kernel, mode="same", boundary="fill", fillvalue=0)

my_result = convolution_two_loops(image, kernel, padding="same")

print("Maximum difference:", np.max(np.abs(my_result - scipy_result)))

#1.1.2: testing on my image 
from skimage.io import imread
import matplotlib.pyplot as plt

image = imread("my_photo.jpg", as_gray=True)
print(image.shape)

#check if my image is normalized and if not cnormalize
image = image.astype(float)
if image.max() > 1:
    image /= 255.0

#display
plt.imshow(image, cmap="gray")
plt.axis("off")

#1.1.3: box filter (9x9)
box_filter = np.ones((9, 9)) / 81
box_result = convolution_two_loops(image, box_filter, padding="same")

#display
plt.imshow(box_result, cmap="gray")
plt.axis("off")

#1.1.4: funite difference
#D_x and D_y
Dx = np.array([[-1, 0, 1]])
Dy = np.array([[-1],[0],[1]])

Dx_result = convolution_two_loops(
    image,
    Dx,
    padding="same"
)

Dy_result = convolution_two_loops(
    image,
    Dy,
    padding="same"
)

#display
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image, cmap="gray")
axes[0].set_title("Original")

axes[1].imshow(Dx_result, cmap="gray")
axes[1].set_title("Dx")

axes[2].imshow(Dy_result, cmap="gray")
axes[2].set_title("Dy")

for ax in axes:
    ax.axis("off")

plt.show()

## Part 1.2 Finite Difference Operator


## Part 1.3 Derivative of Gaussian (DoG) Filter
## [Optional] Bells & WHistles

# Part 2: Fun with Frrequencies!

## Part 2.1: Image "Sharpening"
## Part 2.2: Hybrid Images
## [Optional] Bells & Whistles

# Multi-resolution blending and the oraple journey
## Part 2.3: Gaussian anf leplavoam stacks
## Part 2.4: Multiresolution Blending (a.k.a. the oraple!)
