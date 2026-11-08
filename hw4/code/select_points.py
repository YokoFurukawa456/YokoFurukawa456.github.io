import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Click corresponding points and save them to data/points/<name>.json.
#
# Rectification (4 corners: top-left, top-right, bottom-right, bottom-left):
#   python select_points.py poster poster.JPG
# Mosaic pair (click the same feature in image 1, then image 2, alternating):
#   python select_points.py set2 set2_left.JPG set2_right.JPG
#
# Points are saved in full-resolution pixel coordinates; main.py rescales them.
# A points file for a mosaic replaces the automatic SIFT matches for that pair.

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

name, files = sys.argv[1], sys.argv[2:]
ims = [plt.imread(os.path.join(DATA_DIR, f)) for f in files]

fig, axes = plt.subplots(1, len(ims), figsize=(8 * len(ims), 6), squeeze=False)
for ax, im in zip(axes[0], ims):
    ax.imshow(im)
    ax.axis("off")
fig.suptitle("Click points (alternate images for a pair). Press Enter when done.")

clicks = plt.ginput(n=-1, timeout=0)
plt.close(fig)

if len(ims) == 1:
    out = {"im1": clicks}
else:
    # ginput gives coordinates in whichever axes was clicked; alternate 1, 2, 1, 2, ...
    out = {"im1": clicks[0::2], "im2": clicks[1::2]}

os.makedirs(os.path.join(DATA_DIR, "points"), exist_ok=True)
with open(os.path.join(DATA_DIR, "points", f"{name}.json"), "w") as f:
    json.dump(out, f)
print(f"saved {len(clicks)} points to data/points/{name}.json")
