# CS180 (CS280A): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio

# name of the input file

# JPEGS 
#imname = 'hw1/data/cathedral.jpg'
#imname = 'hw1/data/monastery.jpg'
#imname = 'hw1/data/tobolsk.jpg'

#imname = 'hw1/data/flowers.tif'
imname = 'hw1/data/camel.tif'
#imname = 'hw1/data/capri.tif'

# TIFFS
#imname = 'hw1/data/church.tif'
#imname = 'hw1/data/emir.tif'
#imname = 'hw1/data/harvesters.tif'
#imname = 'hw1/data/icon.tif'
#imname = 'hw1/data/ilemselga.tif'
#imname = 'hw1/data/melons.tif'
#imname = 'hw1/data/religous_painting.tif'
#imname = 'hw1/data/self_portrait.tif'
#imname = 'hw1/data/siren.tif'
#imname = 'hw1/data/three_generations.tif'
#imname = 'hw1/data/wharf.tif'

# read in the image
im = skio.imread(imname)

# convert to double (might want to do this later on to save memory)    
im = sk.img_as_float(im)
    
# compute the height of each part (just 1/3 of total)
height = im.shape[0] // 3

# separate color channels
b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]

# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)

#############
# L2
#############

def overlapping_regions(im1, im2, dx, dy, border_fraction=0.10):
    """Return the central, non-wrapping overlap for an im1 shift of (dx, dy)."""
    if dx > 0:
        x1, x2 = slice(0, -dx), slice(dx, None)
    elif dx < 0:
        x1, x2 = slice(-dx, None), slice(0, dx)
    else:
        x1 = x2 = slice(None)

    if dy > 0:
        y1, y2 = slice(0, -dy), slice(dy, None)
    elif dy < 0:
        y1, y2 = slice(-dy, None), slice(0, dy)
    else:
        y1 = y2 = slice(None)

    a = im1[y1, x1]
    b = im2[y2, x2]

    # Ignore plate borders, which are not reliable image content for alignment.
    border_y = int(a.shape[0] * border_fraction)
    border_x = int(a.shape[1] * border_fraction)
    if border_y > 0 and border_x > 0:
        a = a[border_y:-border_y, border_x:-border_x]
        b = b[border_y:-border_y, border_x:-border_x]

    return a, b


def align(im1, im2, max_shift):
    best_score = np.inf
    best_dx = 0
    best_dy = 0
    for dx in range(-max_shift, max_shift + 1):
        for dy in range(-max_shift, max_shift + 1):
            a, b = overlapping_regions(im1, im2, dx, dy)

            # Normalize each overlap so L2 measures structure, not brightness.
            a = (a - np.mean(a)) / (np.std(a) + 1e-8)
            b = (b - np.mean(b)) / (np.std(b) + 1e-8)
            score = np.mean((a - b) ** 2)
            if score < best_score:
                best_score = score
                best_dx = dx
                best_dy = dy
    return best_dx, best_dy

#############
# NNC
#############

def ncc_align(im1, im2, max_shift, start_dx=0, start_dy=0):

    best_score = -np.inf
    best_dx = start_dx
    best_dy = start_dy

    for dx in range(start_dx - max_shift, start_dx + max_shift + 1):
        for dy in range(start_dy - max_shift, start_dy + max_shift + 1):
            a, b = overlapping_regions(im1, im2, dx, dy)

            # Normalize
            a = a - np.mean(a)
            b = b - np.mean(b)

            denominator = np.linalg.norm(a) * np.linalg.norm(b)

            if denominator == 0:
                continue

            score = np.sum(a * b) / denominator

            if score > best_score:
                best_score = score
                best_dx = dx
                best_dy = dy

    return best_dx, best_dy

# # align the green and red channels to the blue channel
# g_dx, g_dy = ncc_align(g, b, 15)
# r_dx, r_dy  = ncc_align(r, b, 15)

# print("Green offset:", (g_dx, g_dy))
# print("Red offset:", (r_dx, r_dy))

# g_aligned = np.roll(g, (g_dy, g_dx), axis=(0, 1))
# r_aligned = np.roll(r, (r_dy, r_dx), axis=(0, 1))

# margin = 15

# g_aligned = g_aligned[margin:-margin, margin:-margin]
# r_aligned = r_aligned[margin:-margin, margin:-margin]
# b_cropped = b[margin:-margin, margin:-margin]

# im_out = np.dstack([r_aligned, g_aligned, b_cropped])

# # display the image
# skio.imshow(im_out)
# skio.show()

# # save the image
# fname = 'save/out_fname.jpg'
# skio.imsave(fname, im_out)

# # display the image
# skio.imshow(im_out)
# skio.show()


##########
# image pyramid
##########

def build_pyramid(im, levels):
    pyramid = [im]

    for i in range(1, levels):
        smaller = sk.transform.rescale(
            pyramid[-1],
            0.5,
            anti_aliasing=True
        )
        pyramid.append(smaller)

    return pyramid

# coarse to fine alignment

def pyramid_align(im1, im2, levels=4, max_shift=15):

    pyramid1 = build_pyramid(im1, levels)
    pyramid2 = build_pyramid(im2, levels)

    dx = 0
    dy = 0

    for level in range(levels - 1, -1, -1):

        image1 = pyramid1[level]
        image2 = pyramid2[level]

        if level < levels - 1:
            dx *= 2
            dy *= 2

        dx, dy = ncc_align(
            image1,
            image2,
            max_shift,
            dx,
            dy
        )

        print("level:", level, "offset:", (dx, dy))

    return dx, dy


def crop_to_common_overlap(b, g, r, g_dx, g_dy, r_dx, r_dy):
    """Crop rolled channels to the area containing real pixels in every channel."""
    height, width = b.shape

    # A channel shifted by dx is valid from x=dx through x=width+dx.
    x_start = max(0, g_dx, r_dx)
    x_end = min(width, width + g_dx, width + r_dx)
    y_start = max(0, g_dy, r_dy)
    y_end = min(height, height + g_dy, height + r_dy)

    return (
        b[y_start:y_end, x_start:x_end],
        g[y_start:y_end, x_start:x_end],
        r[y_start:y_end, x_start:x_end],
    )


def auto_white_balance(image):
    """Apply gray-world white balance so the three channel averages agree."""
    channel_means = np.mean(image, axis=(0, 1))
    target_mean = np.mean(channel_means)
    scale = target_mean / (channel_means + 1e-8)
    return np.clip(image * scale, 0, 1)

# Edge maps are much less affected by the different brightness responses of
# the blue, green, and red photographic filters.  Find shifts on the edges,
# then apply those shifts to the original channel images below.
from skimage.filters import sobel

b_edges = sobel(b)
g_edges = sobel(g)
r_edges = sobel(r)

g_dx, g_dy = pyramid_align(g_edges, b_edges, levels=4, max_shift=15)
r_dx, r_dy = pyramid_align(r_edges, b_edges, levels=4, max_shift=15)

print("FINAL Green offset:", (g_dx, g_dy))
print("FINAL Red offset:", (r_dx, r_dy))

# Apply final offsets to original-resolution images
g_aligned = np.roll(g, (g_dy, g_dx), axis=(0, 1))
r_aligned = np.roll(r, (r_dy, r_dx), axis=(0, 1))

# Automatically remove every border introduced by channel shifts.
b_cropped, g_aligned, r_aligned = crop_to_common_overlap(
    b, g_aligned, r_aligned, g_dx, g_dy, r_dx, r_dy
)

# Combine channels
im_out = np.dstack([
    r_aligned,
    g_aligned,
    b_cropped
])

# Correct the overall channel color cast after alignment and cropping.
im_out = auto_white_balance(im_out)

# Display
import matplotlib.pyplot as plt

plt.imshow(im_out)
plt.axis('off')
plt.show()
