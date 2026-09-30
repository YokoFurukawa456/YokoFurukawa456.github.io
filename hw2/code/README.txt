CS180 Project 2: Fun with Filters and Frequencies
=================================================

Files
-----
hw2/
  code/
    main.py               all code for Parts 1.1-2.4, in order
    align_image_code.py   provided alignment code for hybrid images
                          (added an optional pts argument to skip clicking)
    README.txt            this file
  data/                   input images
  save/                   output figures (these are what the website shows)
  web/                    index.html + style.css for the write-up


Requirements
------------
Python 3 with:
  numpy, scipy, matplotlib, opencv-python (cv2), scikit-image, Pillow

Install with:
  pip3 install numpy scipy matplotlib opencv-python scikit-image pillow


How to run
----------
Run from the hw2/ folder (NOT from hw2/code/), because all image paths
are relative to hw2/, e.g. "data/cameraman.png" and "save/...":

  cd hw2
  python3 code/main.py

The script runs top to bottom through all parts. Figures are saved to
save/ with plt.savefig, and plt.show() opens a window for some figures.
The script pauses at each plt.show() until you close the window.


Running only some parts
-----------------------
main.py is split into sections with headers like
  ## Part 1.2 Finite Difference Operator ##
Many plotting/saving blocks are commented out so that a full run doesn't
open dozens of windows. To regenerate a figure, uncomment its block
(including its plt.savefig line). If you comment out a plot, also comment
out its plt.savefig line, or it will save a blank image over the old one.

Later parts reuse functions defined earlier (e.g. gaussian_stack and
laplacian_stack from Part 2.3 are used in Part 2.4), so keep the
definitions active even if you skip a part's figures.


Clicking points for alignment (Part 2.2)
----------------------------------------
Hybrid images need the two images aligned. align_images() opens a window
for each image; click 2 points on each (e.g. the two eyes), in the same
order on both images:
  - first window:  the HIGH-frequency image
  - second window: the LOW-frequency image
After clicking, the terminal prints "Alignment points: ...". Paste those
into the pts value (e.g. in the hybrid_pairs list in section 2.2.4) and
later runs will skip the clicking.

The multiresolution blends in Part 2.4 (portal, Thanos stones, Peter /
Spider-Man) use fixed, hand-measured points, so they need no clicking.


What each part produces (in save/)
----------------------------------
Part 1.1  convolution with 4 loops / 2 loops vs scipy (prints runtimes and
          max difference), box filter and D_x / D_y on my photo
Part 1.2  partial derivatives, gradient magnitude, binarized edges
Part 1.3  Gaussian and DoG filters, blurred gradient, DoG edges, and a
          comparison showing blur-then-derivative == DoG (prints max difference)
Part 2.1  unsharp masking: 2.1_taj.png, 2.1_sf.png, 2.1_taj_alpha.png
          (different alpha values), 2.1_taj_resharpen.png (blur then sharpen)
Part 2.2  hybrid images: 2.2_<pair>_aligned.png, 2.2_<pair>_hybrid.png,
          Fourier transform plots
Part 2.3  Gaussian / Laplacian stacks of the apple and orange, reconstruction
Part 2.4  oraple (2.4_oraple*.png), Figure 3.42 recreation (2.4_figure342.png),
          Berkeley through Dr. Strange's portal (2.4_portal*.png),
          Thanos in the Infinity Stones (2.4_thanos_stones*.png; set
          tint_faces = False for the untinted version),
          half Peter / half Spider-Man (2.4_peter_spider*.png)


Notes
-----
- PNGs load with values 0-1; JPGs load as 0-255. The code normalizes to
  0-1 where needed and drops the alpha channel from RGBA PNGs.
- Convolutions on the images use boundary="symm" (mirrored padding) so
  blur-then-derivative and the DoG filter give identical results.
- The website is web/index.html; it links images from ../save/.
