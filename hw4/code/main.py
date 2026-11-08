import numpy as np
import cv2
import os
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CODE_DIR, "..", "data")
POINTS_DIR = os.path.join(DATA_DIR, "points")
OUT_DIR = os.path.join(CODE_DIR, "..", "web", "images")
SCALE = 0.25  # originals are 5712x4284, work at 1428x1071

# Mosaic pairs: (name, reference image, image warped into the reference's frame).
# Add more pairs here; missing files are skipped.
MOSAICS = [
    ("sf", "sf1.JPG", "sf2.JPG"),
    ("sather", "sather1.JPG", "sather2.JPG"),
    ("dorm", "dorm1.JPG", "dorm2.JPG"),
]

# Rectification: (name, image, target aspect ratio width/height).
# Corners (top-left, top-right, bottom-right, bottom-left) are read from
# data/points/<name>.json, made with select_points.py.
RECTIFY = [
    ("poster", "posters.JPG", 8.5 / 11),  # letter-size flyer
    ("tile", "tiles.JPG", 1.0),           # square paving tile
]

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(POINTS_DIR, exist_ok=True)


def load_image(filename):
    im = cv2.imread(os.path.join(DATA_DIR, filename))
    assert im is not None, f"couldn't load {filename}"
    return cv2.resize(im, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)


def save(name, im):
    cv2.imwrite(os.path.join(OUT_DIR, name), np.clip(im, 0, 255).astype(np.uint8),
                [cv2.IMWRITE_JPEG_QUALITY, 90])


#A.1: Shoot the Pictures
def side_by_side(images, gap=12):
    h = min(im.shape[0] for im in images)
    resized = [cv2.resize(im, (round(im.shape[1] * h / im.shape[0]), h)) for im in images]
    spacer = np.full((h, gap, 3), 255, np.uint8)
    row = [resized[0]]
    for im in resized[1:]:
        row += [spacer, im]
    return np.hstack(row)


#A.2: Recover Homographies
#p' = Hp, where H is a 3x3 matrix with 8 degrees of freedom (H[2,2] = 1)
def build_system(im1_pts, im2_pts):
    # Each correspondence (x, y) -> (u, v) gives two rows of Ah = b,
    # with h = [h11 h12 h13 h21 h22 h23 h31 h32]
    A, b = [], []
    for (x, y), (u, v) in zip(im1_pts, im2_pts):
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y])
        b += [u, v]
    return np.array(A, float), np.array(b, float)


def computeH(im1_pts, im2_pts):
    # Homography mapping im1_pts to im2_pts (both Nx2, N >= 4).
    # N > 4 gives an overdetermined system, solved with least squares.
    A, b = build_system(im1_pts, im2_pts)
    h, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.append(h, 1).reshape(3, 3)


def apply_H(H, pts):
    p = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    return p[:, :2] / p[:, 2:]


def auto_correspondences(im1, im2, ratio=0.75, iters=3000, thresh=2.0, seed=180):
    # SIFT matches with Lowe's ratio test, then RANSAC built on computeH
    # to drop wrong matches (and points off the dominant plane, e.g. nearby trees).
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = sift.detectAndCompute(cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY), None)
    matches = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [m for m, n in matches if m.distance < ratio * n.distance]
    pts1 = np.array([k1[m.queryIdx].pt for m in good])
    pts2 = np.array([k2[m.trainIdx].pt for m in good])

    rng = np.random.default_rng(seed)
    best = np.zeros(len(pts1), bool)
    for _ in range(iters):
        idx = rng.choice(len(pts1), 4, replace=False)
        H = computeH(pts1[idx], pts2[idx])
        err = np.linalg.norm(apply_H(H, pts1) - pts2, axis=1)
        inliers = err < thresh
        if inliers.sum() > best.sum():
            best = inliers
    return pts1[best], pts2[best], len(good)


def get_correspondences(name, im1, im2):
    # Hand-clicked points in data/points/<name>.json take priority over auto matching.
    # Saved points are in full-resolution coordinates.
    path = os.path.join(POINTS_DIR, f"{name}.json")
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        return np.array(d["im1"], float) * SCALE, np.array(d["im2"], float) * SCALE, None
    pts1, pts2, n_matches = auto_correspondences(im1, im2)
    with open(path, "w") as f:
        json.dump({"im1": (pts1 / SCALE).tolist(), "im2": (pts2 / SCALE).tolist()}, f)
    return pts1, pts2, n_matches


def plot_correspondences(im1, im2, pts1, pts2, filename, n_show=30):
    idx = np.linspace(0, len(pts1) - 1, min(n_show, len(pts1))).astype(int)
    colors = plt.cm.hsv(np.linspace(0, 1, len(idx), endpoint=False))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    for ax, im, pts in zip(axes, [im1, im2], [pts1, pts2]):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.scatter(pts[idx, 0], pts[idx, 1], c=colors, s=40, edgecolors="white", linewidths=1)
        for k, i in enumerate(idx):
            ax.annotate(str(k + 1), pts[i], color="white", fontsize=8,
                        xytext=(4, -4), textcoords="offset points")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=110)
    plt.close(fig)


#A.3: Warp the Images
#imwarped_nn = warpImageNearestNeighbor(im,H)
#imwarped_bil = warpImageBilinear(im,H)
def bounding_box(H, h, w):
    # Pipe the four corners through H to predict the output extent
    corners = apply_H(H, np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], float))
    x0, y0 = np.floor(corners.min(0)).astype(int)
    x1, y1 = np.ceil(corners.max(0)).astype(int)
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def translate_homography(H, tx, ty):
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]], float)
    return translation_matrix @ H


def inverse_map(im, H, canvas):
    # For every output pixel, find where it comes from in im (inverse warping,
    # so the output has no holes). Pixel (0, 0) is the center of the first pixel.
    if canvas is None:
        canvas = bounding_box(H, *im.shape[:2])
    cx, cy, out_w, out_h = canvas
    H_inv = np.linalg.inv(translate_homography(H, -cx, -cy))

    # all output pixel coords as a 3xN homogeneous array
    xs, ys = np.meshgrid(np.arange(out_w), np.arange(out_h))
    p = np.stack([xs.ravel(), ys.ravel(), np.ones(out_h*out_w)])
    p_prime = H_inv @ p
    behind = p_prime[2] <= 0  # past the horizon: no real source pixel, mark out of bounds
    p_prime /= np.where(behind, 1, p_prime[2])
    p_prime[:2, behind] = -1
    return p_prime[0], p_prime[1], out_h, out_w


def warpImageNearestNeighbor(im, H, canvas=None):
    # Returns a float RGBA image: color channels plus an alpha mask (1 = has a value)
    h, w = im.shape[:2]
    x_prime, y_prime, out_h, out_w = inverse_map(im, H, canvas)

    x_prime = np.round(x_prime).astype(int)
    y_prime = np.round(y_prime).astype(int)
    valid = (0 <= x_prime) & (x_prime < w) & (0 <= y_prime) & (y_prime < h)

    imwarped = np.zeros((out_h * out_w, 4), np.float32)
    imwarped[valid, :3] = im[y_prime[valid], x_prime[valid]]
    imwarped[valid, 3] = 1
    return imwarped.reshape(out_h, out_w, 4)


def warpImageBilinear(im, H, canvas=None):
    h, w = im.shape[:2]
    x_prime, y_prime, out_h, out_w = inverse_map(im, H, canvas)

    valid = (0 <= x_prime) & (x_prime <= w-1) & (0 <= y_prime) & (y_prime <= h-1)
    x_prime, y_prime = x_prime[valid], y_prime[valid]

    # clip so the right/bottom edge pixels still have a neighbor to interpolate with
    x0 = np.minimum(np.floor(x_prime).astype(int), w-2)
    y0 = np.minimum(np.floor(y_prime).astype(int), h-2)
    x1, y1 = x0 + 1, y0 + 1
    dx, dy = (x_prime - x0)[:, None], (y_prime - y0)[:, None]  # [:, None] broadcasts over color channels

    im = im.astype(np.float32)
    imwarped = np.zeros((out_h * out_w, 4), np.float32)
    imwarped[valid, :3] = (im[y0, x0] * (1 - dx) * (1 - dy) +
                           im[y0, x1] * dx * (1 - dy) +
                           im[y1, x0] * (1 - dx) * dy +
                           im[y1, x1] * dx * dy)
    imwarped[valid, 3] = 1
    return imwarped.reshape(out_h, out_w, 4)


def zoom_compare(nn, bil, size=110, zoom=4):
    # Crop the same patch from both warps (where there's the most fine detail) and blow it up
    gray = cv2.cvtColor(bil[..., :3].astype(np.uint8), cv2.COLOR_BGR2GRAY)
    detail = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (0, 0), size / 2)
    inside = cv2.erode(bil[..., 3], np.ones((size, size), np.uint8)) > 0
    cy, cx = np.unravel_index(np.argmax(np.where(inside, detail, 0)), detail.shape)
    crops = [im[cy-size//2:cy+size//2, cx-size//2:cx+size//2, :3] for im in (nn, bil)]
    crops = [np.repeat(np.repeat(c, zoom, 0), zoom, 1).astype(np.uint8) for c in crops]
    return side_by_side(crops)


def rectify(name, filename, aspect, margin=2):
    path = os.path.join(POINTS_DIR, f"{name}.json")
    if not os.path.exists(os.path.join(DATA_DIR, filename)) or not os.path.exists(path):
        print(f"skipping rectification '{name}': need data/{filename} and data/points/{name}.json")
        return
    im = load_image(filename)
    with open(path) as f:
        corners = np.array(json.load(f)["im1"], float) * SCALE  # clicked on the full-res image
    # target rectangle: same width as the clicked top edge, height from the known aspect
    rw = np.linalg.norm(corners[1] - corners[0])
    rh = rw / aspect
    x, y = corners[0]
    target = np.array([[x, y], [x + rw, y], [x + rw, y + rh], [x, y + rh]])
    H = computeH(corners, target)

    # A ground plane stretches without bound toward the horizon, so keep the output
    # to the bounding box clipped to `margin` rectangle sizes around the target
    bx, by, bw, bh = bounding_box(H, *im.shape[:2])
    x0, y0 = max(bx, int(x - margin * rw)), max(by, int(y - margin * rh))
    x1 = min(bx + bw, int(x + (margin + 1) * rw))
    y1 = min(by + bh, int(y + (margin + 1) * rh))
    out = warpImageBilinear(im, H, (x0, y0, x1 - x0, y1 - y0))
    vis = im.copy()
    cv2.polylines(vis, [corners.astype(np.int32)], True, (0, 220, 255), 3)
    save(f"a3_rect_{name}.jpg", side_by_side([vis, out[..., :3]]))
    print(f"rectified '{name}'")


#A.4: Blend the Images into a Mosaic
def distance_weight(alpha):
    # Distance from each valid pixel to the nearest pixel without a value:
    # 1 deep inside the image, falling to 0 at its edges.
    padded = np.pad((alpha > 0).astype(np.uint8), 1)
    return cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]


def masked_blur(im, alpha, sigma):
    # Low-pass that ignores empty pixels, so the black background doesn't bleed in
    num = cv2.GaussianBlur(im * alpha[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(alpha, (0, 0), sigma)
    return num / np.maximum(den, 1e-6)[..., None]


def blend_mosaic(im1, im2, H21, sigma=4):
    # Leave im1 unwarped and warp im2 into its projection on a shared canvas
    h1, w1 = im1.shape[:2]
    bx = bounding_box(np.eye(3), h1, w1)
    b2 = bounding_box(H21, *im2.shape[:2])
    cx, cy = min(bx[0], b2[0]), min(bx[1], b2[1])
    cw = max(bx[0] + bx[2], b2[0] + b2[2]) - cx
    ch = max(bx[1] + bx[3], b2[1] + b2[3]) - cy
    canvas = (cx, cy, cw, ch)

    w1_im = warpImageBilinear(im1, np.eye(3), canvas)
    w2_im = warpImageBilinear(im2, H21, canvas)
    c1, a1 = w1_im[..., :3], w1_im[..., 3]
    c2, a2 = w2_im[..., :3], w2_im[..., 3]

    # naive overwrite, for comparison
    overwrite = np.where(a2[..., None] > 0, c2, c1)

    # feathering: weighted average with distance-transform alphas
    d1, d2 = distance_weight(a1), distance_weight(a2)
    total = np.maximum(d1 + d2, 1e-6)
    wt1, wt2 = (d1 / total)[..., None], (d2 / total)[..., None]
    feather = wt1 * c1 + wt2 * c2

    # 2-level Laplacian pyramid: feather the low frequencies, but take the high
    # frequencies from whichever image is more "central" there to avoid ghosting
    low1, low2 = masked_blur(c1, a1, sigma), masked_blur(c2, a2, sigma)
    high1, high2 = c1 - low1, c2 - low2
    low = wt1 * low1 + wt2 * low2
    high = np.where((d1 >= d2)[..., None], high1, high2)
    pyramid = (low + high) * ((a1 + a2) > 0)[..., None]

    # close-up on the seam (where both images have equal weight), for comparing blends;
    # pick the most detailed spot along it so the crop isn't flat sky
    s = min(160, ch // 2, cw // 2)
    ys, xs = np.nonzero((a1 > 0) & (a2 > 0) & (np.abs(d1 - d2) < 2))
    gray = cv2.cvtColor(np.clip(pyramid, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    detail = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (0, 0), s / 2)
    k = np.argmax(detail[ys, xs])
    row = int(np.clip(ys[k], s, ch - s))
    col = int(np.clip(xs[k], s, cw - s))
    crops = [np.clip(im[row-s:row+s, col-s:col+s], 0, 255).astype(np.uint8)
             for im in (overwrite, feather, pyramid)]
    return overwrite, feather, pyramid, side_by_side(crops)


results = {}

for name, f1, f2 in MOSAICS:
    if not all(os.path.exists(os.path.join(DATA_DIR, f)) for f in (f1, f2)):
        print(f"skipping mosaic '{name}': missing {f1} or {f2}")
        continue
    image1, image2 = load_image(f1), load_image(f2)

    #A.1
    save(f"a1_{name}_sources.jpg", side_by_side([image1, image2]))

    #A.2: H21 maps image2 points into image1 (the reference)
    pts1, pts2, n_matches = get_correspondences(name, image1, image2)
    H21 = computeH(pts2, pts1)
    err = np.linalg.norm(apply_H(H21, pts2) - pts1, axis=1)
    plot_correspondences(image1, image2, pts1, pts2, f"a2_{name}_points.jpg")
    A, b = build_system(pts2[:3], pts1[:3])

    #A.3
    t = time.perf_counter()
    im2warped_nn = warpImageNearestNeighbor(image2, H21)
    t_nn = time.perf_counter() - t
    t = time.perf_counter()
    im2warped_bil = warpImageBilinear(image2, H21)
    t_bil = time.perf_counter() - t
    save(f"a3_{name}_nn.jpg", im2warped_nn[..., :3])
    save(f"a3_{name}_bil.jpg", im2warped_bil[..., :3])
    save(f"a3_{name}_zoom.jpg", zoom_compare(im2warped_nn, im2warped_bil))

    #A.4
    overwrite, feather, pyramid, seam = blend_mosaic(image1, image2, H21)
    save(f"a4_{name}_seam.jpg", seam)
    save(f"a4_{name}_overwrite.jpg", overwrite)
    save(f"a4_{name}_feather.jpg", feather)
    save(f"a4_{name}_mosaic.jpg", pyramid)

    results[name] = {
        "image_size": list(image1.shape[:2]),
        "sift_matches": n_matches,
        "correspondences": len(pts1),
        "reprojection_error_px": {"mean": float(err.mean()), "max": float(err.max())},
        "H": H21.tolist(),
        "first_rows_A": A.tolist(),
        "first_rows_b": b.tolist(),
        "warp_seconds": {"nearest": t_nn, "bilinear": t_bil},
        "warp_pixels": int(im2warped_nn.shape[0] * im2warped_nn.shape[1]),
    }
    print(f"mosaic '{name}': {len(pts1)} correspondences, "
          f"mean reprojection error {err.mean():.2f}px, "
          f"nn {t_nn:.2f}s, bilinear {t_bil:.2f}s")

for name, filename, aspect in RECTIFY:
    rectify(name, filename, aspect)

with open(os.path.join(CODE_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)


############
## Part B ##
############

# B.1 Harris Corner Detection
# Uses the provided harris.py when scikit-image is installed. Otherwise falls back
# to the cv2/numpy rewrite below, which does the same thing: skimage
# corner_harris(method='eps', sigma=1) + peak_local_max(min_distance=1) + edge discard.
def get_harris_corners_cv2(im, edge_discard=20, sigma=1, eps=1e-6):
    # im: grayscale float image. Returns the Harris response h (same size as im)
    # and corner coords as a 2xN array of (row, col).
    assert edge_discard >= 20  # the 40x40 descriptor windows (B.2) need this margin
    Ix = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=3) / 8
    Iy = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=3) / 8
    # second-moment matrix, smoothed with a Gaussian window
    Sxx = cv2.GaussianBlur(Ix * Ix, (0, 0), sigma)
    Syy = cv2.GaussianBlur(Iy * Iy, (0, 0), sigma)
    Sxy = cv2.GaussianBlur(Ix * Iy, (0, 0), sigma)
    # corner strength f_HM = det(H) / tr(H)  (harmonic mean of eigenvalues, Brown eq. 2)
    h = (Sxx * Syy - Sxy ** 2) / (Sxx + Syy + eps)

    # local maxima in a 3x3 neighborhood (min_distance=1)
    is_peak = (h == cv2.dilate(h, np.ones((3, 3), np.uint8))) & (h > 0)  # drop flat/edge regions (f <= 0)
    rows, cols = np.nonzero(is_peak)

    # discard points near the border
    H, W = im.shape
    keep = ((rows >= edge_discard) & (rows < H - edge_discard) &
            (cols >= edge_discard) & (cols < W - edge_discard))
    return h, np.vstack([rows[keep], cols[keep]])


def dist2_np(x, c):
    # Squared distances between every row of x (MxD) and every row of c (NxD) -> MxN
    return ((x ** 2).sum(1)[:, None] + (c ** 2).sum(1)[None, :] - 2 * x @ c.T).clip(min=0)


try:
    from harris import get_harris_corners, dist2
    HARRIS_SOURCE = "harris.py (skimage)"
except ImportError:
    get_harris_corners, dist2 = get_harris_corners_cv2, dist2_np
    HARRIS_SOURCE = "cv2 fallback (scikit-image not installed)"
print(f"B.1 using Harris from {HARRIS_SOURCE}")


def anms(coords, h, n_keep=500, c_robust=0.9, chunk=2048):
    # Adaptive Non-Maximal Suppression (Brown et al., Section 3).
    # Suppression radius of point i:
    #     r_i = min_j |x_i - x_j|  over all j with  f(x_i) < c_robust * f(x_j)
    # i.e. the distance to the nearest point that is "sufficiently stronger".
    # Keeping the n_keep largest radii gives strong corners spread over the image.
    pts = coords.T.astype(np.float64)        # N x 2, (row, col)
    f = h[coords[0], coords[1]]
    order = np.argsort(-f)                   # strongest first
    pts, f = pts[order], f[order]

    # With f sorted descending, the points that dominate point i (f_j > f_i / c)
    # are exactly a prefix of the list, so compare each point to that prefix only.
    n_dominating = np.searchsorted(-f * c_robust, -f[:], side="left")
    radii = np.full(len(f), np.inf)          # the global max is never suppressed
    for start in range(0, len(f), chunk):
        stop = min(start + chunk, len(f))
        m = n_dominating[start:stop].max()
        if m == 0:
            continue
        d = dist2(pts[start:stop], pts[:m])                       # chunk x m
        d[np.arange(m)[None, :] >= n_dominating[start:stop, None]] = np.inf
        radii[start:stop] = d.min(1)

    best = np.argsort(-radii, kind="stable")[:n_keep]
    return pts[best].astype(int).T, np.sqrt(np.maximum(radii[best], 0))         # 2 x n_keep (row, col)


def plot_corners(im, coords, filename, title=None, s=6, color="red"):
    fig, ax = plt.subplots(figsize=(10, 10 * im.shape[0] / im.shape[1]))
    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    ax.scatter(coords[1], coords[0], s=s, c=color, edgecolors="none")
    if title:
        ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=110)
    plt.close(fig)


def to_gray(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255


# B.2 Feature Descriptor Extraction
# Axis-aligned version of the MOPS descriptor (Brown et al., Section 4): an 8x8 patch
# sampled with spacing 5 from the 40x40 window around each corner. Blurring first
# makes each sample stand for its 5x5 cell (no aliasing) and makes the descriptor
# tolerant of small errors in corner location. Bias/gain normalization makes it
# insensitive to brightness and contrast changes. (No rotation, no wavelets.)
def extract_descriptors(gray, coords, window=40, size=8, sigma=2.5):
    # gray: float grayscale image; coords: 2xN (row, col), at least window/2 from the border.
    # Returns N x 64 normalized descriptors.
    spacing = window // size                                  # 5
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    # sample at the center of each 5x5 cell of the 40x40 window: -18, -13, ..., 17
    offsets = np.arange(size) * spacing - window // 2 + spacing // 2
    rows = coords[0][:, None, None] + offsets[None, :, None]  # N x 8 x 1
    cols = coords[1][:, None, None] + offsets[None, None, :]  # N x 1 x 8
    patches = blurred[rows, cols].reshape(len(coords[0]), -1)  # N x 64

    # bias/gain normalization: zero mean, unit standard deviation
    patches = patches - patches.mean(1, keepdims=True)
    return patches / (patches.std(1, keepdims=True) + 1e-8)


def plot_descriptors(im, coords, desc, filename, n_show=6, window=40, seed=0):
    # Top row: where the features are. Below: each 40x40 window and its 8x8 descriptor.
    rng = np.random.default_rng(seed)
    idx = rng.choice(coords.shape[1], n_show, replace=False)
    colors = plt.cm.tab10(np.arange(n_show))
    half = window // 2

    fig = plt.figure(figsize=(2.2 * n_show, 2.2 * n_show * im.shape[0] / im.shape[1] + 4.6))
    gs = fig.add_gridspec(3, n_show, height_ratios=[n_show * im.shape[0] / im.shape[1], 1, 1])
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    for k, i in enumerate(idx):
        r, c = coords[:, i]
        ax.add_patch(plt.Rectangle((c - half, r - half), window, window,
                                   fill=False, ec=colors[k], lw=2))
        ax.annotate(str(k + 1), (c + half, r - half), color=colors[k], fontsize=11,
                    fontweight="bold", xytext=(2, 0), textcoords="offset points")
    ax.axis("off")

    for k, i in enumerate(idx):
        r, c = coords[:, i]
        a = fig.add_subplot(gs[1, k])
        a.imshow(cv2.cvtColor(im[r-half:r+half, c-half:c+half], cv2.COLOR_BGR2RGB))
        a.set_title(f"{k + 1}: 40x40 window", fontsize=9, color=colors[k])
        a.set_xticks([]); a.set_yticks([])
        a = fig.add_subplot(gs[2, k])
        a.imshow(desc[i].reshape(8, 8), cmap="gray", interpolation="nearest")
        a.set_title("8x8 descriptor", fontsize=9)
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=110)
    plt.close(fig)


def detect_and_describe(im, n_keep=1000):
    gray = to_gray(im)
    h, corners = get_harris_corners(gray)
    t = time.perf_counter()
    chosen, radii = anms(corners, h, n_keep=n_keep)
    t_anms = time.perf_counter() - t
    desc = extract_descriptors(gray, chosen)
    return corners, chosen, radii, t_anms, desc


# B.3 Feature Matching
# Nearest-neighbor matching on descriptor SSD with Lowe's ratio test (Brown et al.,
# Section 5 / Figure 6b): a match is kept only if its best SSD is much smaller than
# the second best, e_1NN / e_2NN < ratio. In Fig. 6b correct matches sit almost
# entirely below ~0.4 while incorrect ones spread around 0.6-1.0.
def match_features(desc1, desc2, ratio=0.4):
    # Returns (idx1, idx2, ratios) for matches desc1[idx1[k]] <-> desc2[idx2[k]].
    d = dist2(desc1, desc2)                      # SSD errors, N1 x N2
    nn = np.argsort(d, axis=1)[:, :2]
    rows = np.arange(len(desc1))
    e1, e2 = d[rows, nn[:, 0]], d[rows, nn[:, 1]]
    r = e1 / np.maximum(e2, 1e-12)
    keep = r < ratio
    return rows[keep], nn[keep, 0], r[keep]


def to_xy(coords):
    # 2xN (row, col) -> Nx2 (x, y), the convention computeH uses
    return coords[::-1].T.astype(float)


def plot_matches(im1, im2, pts1, pts2, filename, title=None, inliers=None, gap=20):
    # Images side by side with a line per match. With `inliers`, outliers are drawn red.
    h = max(im1.shape[0], im2.shape[0])
    canvas = np.full((h, im1.shape[1] + gap + im2.shape[1], 3), 255, np.uint8)
    canvas[:im1.shape[0], :im1.shape[1]] = im1
    canvas[:im2.shape[0], im1.shape[1] + gap:] = im2
    off = im1.shape[1] + gap

    fig, ax = plt.subplots(figsize=(14, 14 * h / canvas.shape[1]))
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    if inliers is None:
        colors = plt.cm.hsv(np.linspace(0, 1, len(pts1), endpoint=False))
    else:
        colors = np.where(inliers[:, None], [[0.1, 0.9, 0.2, 1]], [[1, 0.1, 0.1, 1]])
    for (x1, y1), (x2, y2), c in zip(pts1, pts2, colors):
        ax.plot([x1, x2 + off], [y1, y2], color=c, lw=0.8, alpha=0.8)
    ax.scatter(pts1[:, 0], pts1[:, 1], s=10, c=colors)
    ax.scatter(pts2[:, 0] + off, pts2[:, 1], s=10, c=colors)
    if title:
        ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=110)
    plt.close(fig)


# B.4 RANSAC
# 4-point RANSAC: fit H exactly to 4 random matches, count the matches it maps to
# within `thresh` px, keep the largest consensus set, then refit H to all of its
# inliers with least squares (computeH from Part A).
def ransac_homography(src, dst, iters=5000, thresh=3.0, seed=0):
    # Robust H with dst ~ H src. Returns (H, inlier mask).
    assert len(src) >= 4, "need at least 4 matches"
    rng = np.random.default_rng(seed)
    best = np.zeros(len(src), bool)
    for _ in range(iters):
        idx = rng.choice(len(src), 4, replace=False)
        H = computeH(src[idx], dst[idx])
        err = np.linalg.norm(apply_H(H, src) - dst, axis=1)
        inliers = err < thresh
        if inliers.sum() > best.sum():
            best = inliers
    return computeH(src[best], dst[best]), best


def manual_homography(name):
    # Hand-clicked points (select_points.py <name>_manual ...) if present; otherwise the
    # Part A correspondences in data/points/<name>.json (SIFT + RANSAC in Part A).
    for fname, label in [(f"{name}_manual.json", "manual (clicked)"),
                         (f"{name}.json", "Part A (SIFT)")]:
        path = os.path.join(POINTS_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            p1, p2 = np.array(d["im1"], float) * SCALE, np.array(d["im2"], float) * SCALE
            return computeH(p2, p1), label, p1, p2
    return None, None, None, None


def plot_mosaic_comparison(left, right, titles, filename):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax, im, t in zip(axes, [left, right], titles):
        ax.imshow(cv2.cvtColor(np.clip(im, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        ax.set_title(t)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=110)
    plt.close(fig)


def reference_error(H, p1, p2):
    # Mean error (px) of H on the reference correspondences p2 -> p1. Measured on the
    # points themselves, i.e. inside the overlap, not extrapolated to the image corners.
    return float(np.linalg.norm(apply_H(H, p2) - p1, axis=1).mean())


results_b = {}
for name, f1, f2 in MOSAICS:
    if not all(os.path.exists(os.path.join(DATA_DIR, f)) for f in (f1, f2)):
        continue
    image1, image2 = load_image(f1), load_image(f2)
    corners1, chosen1, radii1, t_anms, desc1 = detect_and_describe(image1)
    corners2, chosen2, radii2, _, desc2 = detect_and_describe(image2)

    #B.1
    plot_corners(image1, corners1, f"b1_{name}_harris.jpg",
                 f"All Harris corners ({corners1.shape[1]})", s=2)
    plot_corners(image1, chosen1, f"b1_{name}_anms.jpg",
                 f"ANMS: {chosen1.shape[1]} corners", s=14)
    print(f"B.1 '{name}': {corners1.shape[1]} Harris corners -> {chosen1.shape[1]} after ANMS "
          f"(min radius {radii1[-1]:.1f}px, {t_anms:.2f}s)")

    #B.2
    plot_descriptors(image1, chosen1, desc1, f"b2_{name}_descriptors.jpg")

    #B.3
    i1, i2, ratios = match_features(desc1, desc2)
    pts1, pts2 = to_xy(chosen1)[i1], to_xy(chosen2)[i2]
    plot_matches(image1, image2, pts1, pts2, f"b3_{name}_matches.jpg",
                 f"{len(i1)} matches after the ratio test (e1/e2 < 0.4)")

    #B.4: H21 maps image2 points into image1 (the reference), as in Part A
    H_auto, inl = ransac_homography(pts2, pts1)
    err = np.linalg.norm(apply_H(H_auto, pts2[inl]) - pts1[inl], axis=1)
    plot_matches(image1, image2, pts1, pts2, f"b4_{name}_ransac.jpg",
                 f"RANSAC: {inl.sum()} inliers (green), {(~inl).sum()} outliers (red)",
                 inliers=inl)
    _, _, auto_mosaic, _ = blend_mosaic(image1, image2, H_auto)
    save(f"b4_{name}_auto_mosaic.jpg", auto_mosaic)

    H_man, man_label, ref1, ref2 = manual_homography(name)
    res = {"harris_corners": int(corners1.shape[1]),
           "anms_corners": int(chosen1.shape[1]),
           "anms_min_radius_px": float(radii1[-1]),
           "anms_seconds": t_anms,
           "ratio_test_matches": int(len(i1)),
           "ransac_inliers": int(inl.sum()),
           "inlier_error_px": {"mean": float(err.mean()), "max": float(err.max())},
           "H_auto": H_auto.tolist()}
    msg = ""
    if H_man is not None:
        _, _, man_mosaic, _ = blend_mosaic(image1, image2, H_man)
        plot_mosaic_comparison(man_mosaic, auto_mosaic,
                               [f"{man_label}: {len(ref1)} correspondences",
                                f"Automatic (Harris + ANMS + MOPS + RANSAC): {inl.sum()} inliers"],
                               f"b4_{name}_compare.jpg")
        e_man, e_auto = reference_error(H_man, ref1, ref2), reference_error(H_auto, ref1, ref2)
        res.update({"compared_with": man_label,
                    "error_on_reference_points_px": {"reference_H": e_man, "auto_H": e_auto}})
        msg = (f", on the {len(ref1)} {man_label} points: reference H {e_man:.2f}px, "
               f"auto H {e_auto:.2f}px")
    results_b[name] = res
    print(f"B.3/B.4 '{name}': {len(i1)} matches -> {inl.sum()} RANSAC inliers, "
          f"inlier error {err.mean():.2f}px{msg}")

with open(os.path.join(CODE_DIR, "results_b.json"), "w") as f:
    json.dump(results_b, f, indent=2)
