import argparse
import time
from PIL import Image
import numpy as np
import sys
from collections import deque, defaultdict
import math
import random

parser = argparse.ArgumentParser(description="Image processing tool: Morphology and Segmentation")

# General Arguments
parser.add_argument('--command', type=str, required=True, 
                    help="Command to run: dilation, erosion, opening, closing, hmt, m6, region_growing")
parser.add_argument('--input', type=str, required=True, help="Input BMP image file")
parser.add_argument('--output', type=str, required=True, help="Output BMP image file")

# Morphology Arguments
parser.add_argument('--se_shape', type=str, default='cross', choices=['cross', 'square'], 
                    help="Shape of the basic structural element")
parser.add_argument('--se_size', type=int, default=3, help="Size of the structural element (e.g., 3 for 3x3)")

# Segmentation Arguments
parser.add_argument('--seed_x', type=int, help="X coordinate for seed point")
parser.add_argument('--seed_y', type=int, help="Y coordinate for seed point")
parser.add_argument('--threshold', type=int, default=10, help="Threshold for region growing")

# Additional R1 options
parser.add_argument('--seeds', type=str, default=None,
                    help="Manual seeds as semicolon-separated row,col pairs: 'r1,c1;r2,c2'")
parser.add_argument('--auto_seeds', action='store_true',
                    help="Automatically select seeds using grid sampling")
parser.add_argument('--grid', type=int, default=50,
                    help="Distance (pixels) between automatic seeds")
parser.add_argument('--adjacency', type=int, default=8, choices=[4, 8],
                    help="Neighbourhood type for region growing (4 or 8)")
parser.add_argument('--min_region_size', type=int, default=50,
                    help="Minimum region size for merging small regions")
parser.add_argument('--merge_threshold', type=float, default=10.0,
                    help="Intensity (or color distance) threshold to merge neighboring regions")


args = parser.parse_args()


# --- IO Functions with Error Handling ---

def read_image(input_file, as_binary=False):
    try:
        # Convert to Grayscale
        img = Image.open(input_file).convert('L')
        arr = np.array(img)
        
        if as_binary:
            # Threshold: everything > 127 becomes 1, else 0
            arr = (arr > 127).astype(np.uint8)
            
        return arr
    except Exception as e:
        print(f"Error reading image: {e}")
        return None


def save_image(output_file, image_array, is_binary=False):
    try:
        # If binary (0/1), scale to 0/255 for visibility
        if is_binary:
            image_array = image_array * 255
            
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        Image.fromarray(image_array).save(output_file)
        print(f"Image saved as {output_file}")
    except Exception as e:
        print(f"Error saving image: {e}")


def get_structural_element(shape, size):
    """ Generates the mask (SE) for basic operations """
    if size % 2 == 0:
        size += 1 
        
    se = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    
    if shape == 'square':
        se[:] = 1
    elif shape == 'cross':
        se[center, :] = 1
        se[:, center] = 1
        
    return se


# --- Morphological Operations ---

def dilation(image, se):
    h, w = image.shape
    se_h, se_w = se.shape
    pad_h, pad_w = se_h // 2, se_w // 2
    
    # Pad with 0 (Background)
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output = np.zeros_like(image)
    
    for i in range(h):
        for j in range(w):
            roi = padded[i:i+se_h, j:j+se_w]
            # If any pixel overlaps, set to 1
            if np.sum(roi * se) > 0:
                output[i, j] = 1
    return output


def erosion(image, se):
    h, w = image.shape
    se_h, se_w = se.shape
    pad_h, pad_w = se_h // 2, se_w // 2
    
    # Pad with 1 (Foreground)
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=1)
    output = np.zeros_like(image)
    
    se_sum = np.sum(se)
    
    for i in range(h):
        for j in range(w):
            roi = padded[i:i+se_h, j:j+se_w]
            # If fit is perfect, set to 1
            if np.sum(roi * se) == se_sum:
                output[i, j] = 1
    return output


def opening(image, se):
    return dilation(erosion(image, se), se)


def closing(image, se):
    return erosion(dilation(image, se), se)


def hit_or_miss(image, b_fg, b_bg):
    match_fg = erosion(image, b_fg)
    
    image_inv = 1 - image
    match_bg = erosion(image_inv, b_bg)

    return match_fg & match_bg


# --- Task M6: Thickening Implementation ---

def get_golay_masks_m6():
    """
    Generates the 8 structural elements for Task M6.
    Based on (xii) but 1 and 0 are swapped.
    """
    masks = []
    
    # Base "North" Mask (Swapped values from standard thinning)
    # Original (xii) North: Top=0, Bottom=1
    # Swapped for M6: Top=1 (FG), Bottom=0 (BG), Middle=Don't Care
    
    # 1 = Must be White
    # 0 = Must be Black
    # -1 = Don't Care
    
    base_mask = np.array([
        [ 1,  1,  1],
        [-1, -1, -1],
        [ 0,  0,  0]
    ], dtype=np.int8)

    # Base "North-East" Mask
    mask_ne = np.array([
        [-1,  1,  1],
        [ 0, -1,  1],
        [ 0,  0, -1]
    ], dtype=np.int8)

    raw_masks = []
    
    # Generate N, E, S, W rotations
    curr = base_mask
    for _ in range(4):
        raw_masks.append(curr)
        curr = np.rot90(curr, -1) # 90 deg clockwise
        
    # Generate NE, SE, SW, NW rotations
    curr = mask_ne
    for _ in range(4):
        raw_masks.append(curr)
        curr = np.rot90(curr, -1)
        
    # Reorder to N, NE, E, SE, S, SW, W, NW
    ordered_masks = [
        raw_masks[0], raw_masks[4], raw_masks[1], raw_masks[5],
        raw_masks[2], raw_masks[6], raw_masks[3], raw_masks[7]
    ]

    # Convert to (FG, BG) pairs for Hit-Or-Miss
    for m in ordered_masks:
        fg = (m == 1).astype(np.uint8)
        bg = (m == 0).astype(np.uint8)
        masks.append((fg, bg))
        
    return masks

def m6_thickening(image):
    """
    Iterative thickening.
    C(A, B) = A U (A HMT B) for all 8 masks until convergence.
    """
    masks = get_golay_masks_m6()
    current_image = image.copy()
    iteration = 0
    
    while True:
        iteration += 1
        previous_image = current_image.copy()
        
        for (b_fg, b_bg) in masks:
            # 1. Find pixels that match the pattern
            hmt_res = hit_or_miss(current_image, b_fg, b_bg)
            
            # 2. Union with original (Add pixels)
            current_image = current_image | hmt_res
            
        # Check if image stopped changing
        if np.array_equal(current_image, previous_image):
            print(f"M6 Converged after {iteration} iterations.")
            break
            
    return current_image
    
# ------------------- Region Growing Helpers -------------------

def parse_seeds_arg(seeds_arg):
    """Parse string like 'r1,c1;r2,c2' -> list of (row,col) tuples."""
    seeds = []
    if not seeds_arg:
        return seeds
    parts = seeds_arg.split(';')
    for p in parts:
        try:
            r_str, c_str = p.split(',')
            seeds.append((int(r_str), int(c_str)))
        except Exception:
            print(f"Warning: invalid seed '{p}', expected format 'r,c'")
    return seeds


def get_neighbors(x, y, h, w, adjacency):
    """Return list of neighbor coordinates (4 or 8 adjacency)."""
    if adjacency == 4:
        offsets = [(1,0), (-1,0), (0,1), (0,-1)]
    else:  # 8 adjacency
        offsets = [(1,0), (-1,0), (0,1), (0,-1),
                   (1,1), (1,-1), (-1,1), (-1,-1)]
    neighbors = []
    for dx, dy in offsets:
        nx, ny = x + dx, y + dy
        if 0 <= nx < h and 0 <= ny < w:
            neighbors.append((nx, ny))
    return neighbors


def auto_seed_selection(image, grid=50):
    """Generate seeds at grid-like intervals across the image."""
    seeds = []
    h, w = image.shape[:2]
    for i in range(0, h, grid):
        for j in range(0, w, grid):
            seeds.append((i, j))
    return seeds


def labelmap_to_color(labels):
    """Convert label map (int labels) to a color image for visualization."""
    h, w = labels.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)

    unique_labels = np.unique(labels)
    random.seed(0)

    colors = {label: (random.randint(0,255),
                      random.randint(0,255),
                      random.randint(0,255))
              for label in unique_labels if label != 0}

    for label, color in colors.items():
        output[labels == label] = color

    # keep background black
    return output


def region_growing_dynamic(image, seeds, threshold, adjacency=8):
    """
    Region growing that updates region mean on the fly.
    - image: ndarray (H,W) grayscale or (H,W,3) color
    - seeds: list of (row, col)
    Returns: labels (H,W int32), num_regions, region_stats
    """
    h, w = image.shape[:2]
    labels = np.zeros((h, w), dtype=np.int32)
    region_stats = {}  # label -> {'sum': scalar or [r,g,b], 'count': int, 'mean': scalar or [r,g,b]}
    current_label = 1

    for seed in seeds:
        sx, sy = seed
        if not (0 <= sx < h and 0 <= sy < w):
            continue
        if labels[sx, sy] != 0:
            continue

        # initialize region with the seed pixel
        if image.ndim == 2:
            ssum = int(image[sx, sy])
            smean = float(ssum)
        else:
            ssum = [int(v) for v in image[sx, sy]]
            smean = [float(v) for v in ssum]

        region_stats[current_label] = {'sum': ssum, 'count': 1, 'mean': smean}
        labels[sx, sy] = current_label

        q = deque()
        q.append((sx, sy))

        while q:
            x, y = q.popleft()
            for nx, ny in get_neighbors(x, y, h, w, adjacency):
                if labels[nx, ny] != 0:
                    continue
                # distance between neighbor pixel and current region mean
                if image.ndim == 2:
                    dist = abs(int(image[nx, ny]) - region_stats[current_label]['mean'])
                else:
                    mean = region_stats[current_label]['mean']
                    px = image[nx, ny]
                    dist = math.sqrt(sum((mean[i] - int(px[i]))**2 for i in range(3)))

                if dist <= threshold:
                    labels[nx, ny] = current_label
                    q.append((nx, ny))

                    # update region stats
                    if image.ndim == 2:
                        region_stats[current_label]['sum'] += int(image[nx, ny])
                        region_stats[current_label]['count'] += 1
                        region_stats[current_label]['mean'] = region_stats[current_label]['sum'] / region_stats[current_label]['count']
                    else:
                        region_stats[current_label]['sum'] = [region_stats[current_label]['sum'][i] + int(image[nx, ny][i]) for i in range(3)]
                        region_stats[current_label]['count'] += 1
                        region_stats[current_label]['mean'] = [region_stats[current_label]['sum'][i] / region_stats[current_label]['count'] for i in range(3)]

        current_label += 1

    return labels, current_label - 1, region_stats


def merge_small_regions(labels, image, min_size=50, merge_threshold=10.0, adjacency=8):
    """
    Merge regions smaller than min_size into the most similar adjacent region
    if mean difference <= merge_threshold. Returns (labels, merged_flag).
    """
    h, w = labels.shape
    # compute region pixels
    region_pixels = defaultdict(list)
    for i in range(h):
        for j in range(w):
            lab = labels[i, j]
            if lab != 0:
                region_pixels[lab].append((i, j))

    # compute region means
    region_mean = {}
    for lab, pix_list in region_pixels.items():
        if image.ndim == 2:
            vals = [int(image[x, y]) for (x, y) in pix_list]
            region_mean[lab] = sum(vals) / len(vals)
        else:
            sums = [0, 0, 0]
            for (x, y) in pix_list:
                p = image[x, y]
                for k in range(3):
                    sums[k] += int(p[k])
            region_mean[lab] = [s / len(pix_list) for s in sums]

    small_regions = [lab for lab, pix in region_pixels.items() if len(pix) < min_size]
    merged_any = False

    for lab in small_regions:
        # find neighboring labels and border counts
        neighbor_counts = {}
        for (x, y) in region_pixels[lab]:
            for nx, ny in get_neighbors(x, y, h, w, adjacency):
                nlab = labels[nx, ny]
                if nlab != 0 and nlab != lab:
                    neighbor_counts.setdefault(nlab, 0)
                    neighbor_counts[nlab] += 1
        if not neighbor_counts:
            continue
        # pick neighbor with largest border contact
        best_neighbor = max(neighbor_counts.items(), key=lambda kv: kv[1])[0]

        # compute mean difference
        def mean_diff(a, b):
            if isinstance(a, list):
                return math.sqrt(sum((a[i] - b[i])**2 for i in range(3)))
            else:
                return abs(a - b)

        if mean_diff(region_mean[lab], region_mean[best_neighbor]) <= merge_threshold:
            for (x, y) in region_pixels[lab]:
                labels[x, y] = best_neighbor
            merged_any = True

    return labels, merged_any


# --- Main Execution Block ---

if args.command in ['dilation', 'erosion', 'opening', 'closing', 'hmt', 'm6']:
    image = read_image(args.input, as_binary=True)
    if image is None: sys.exit(1)
    
    start = time.time()
    output_image = None
    
    if args.command == 'dilation':
        se = get_structural_element(args.se_shape, args.se_size)
        output_image = dilation(image, se)

    elif args.command == 'erosion':
        se = get_structural_element(args.se_shape, args.se_size)
        output_image = erosion(image, se)

    elif args.command == 'opening':
        se = get_structural_element(args.se_shape, args.se_size)
        output_image = opening(image, se)

    elif args.command == 'closing':
        se = get_structural_element(args.se_shape, args.se_size)
        output_image = closing(image, se)

    elif args.command == 'hmt':
        # Demo: Corner detection
        # FG: Center and Right are 1
        b_fg = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]], dtype=np.uint8)
        # BG: Left is 0
        b_bg = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=np.uint8)
        output_image = hit_or_miss(image, b_fg, b_bg)
        
    elif args.command == 'm6':
        print("Running M6: Thickening...")
        output_image = m6_thickening(image)

    end = time.time()
    print(f"Time: {end - start:.4f}s")
    
    if output_image is not None:
        save_image(args.output, output_image, is_binary=True)

elif args.command == 'region_growing':
    # Load image (preserve color if present)
    try:
        pil = Image.open(args.input)
    except Exception as e:
        print(f"Error opening image: {e}")
        sys.exit(1)

    # Convert to numpy array; keep RGB if possible
    img_arr = np.array(pil)
    # If image has alpha channel, drop it
    if img_arr.ndim == 3 and img_arr.shape[2] == 4:
        img_arr = img_arr[:, :, :3]

    start = time.time()

    # Seed selection
    if args.auto_seeds:
        seeds = auto_seed_selection(img_arr, grid=args.grid)
        print(f"Auto-selected {len(seeds)} seeds (grid={args.grid})")
    else:
        seeds = parse_seeds_arg(args.seeds)
        if not seeds:
            if args.seed_x is not None and args.seed_y is not None:
                seeds = [(args.seed_x, args.seed_y)]
            else:
                print("Error: specify --seeds or --seed_x/--seed_y or use --auto_seeds")
                sys.exit(1)

    print(f"Using {len(seeds)} seed(s). Adjacency={args.adjacency}, threshold={args.threshold}")

    labels, num_regions, stats = region_growing_dynamic(img_arr, seeds, args.threshold, adjacency=args.adjacency)
    print(f"Initial regions: {num_regions}")

    # Merge small regions iteratively until stable
    merged = True
    merge_pass = 0
    while merged:
        merge_pass += 1
        labels, merged = merge_small_regions(labels, img_arr, min_size=args.min_region_size,
                                             merge_threshold=args.merge_threshold, adjacency=args.adjacency)
        if merged:
            print(f"Merge pass {merge_pass}: merged small regions, repeating...")

    unique_labels = np.unique(labels)
    num_final = len(unique_labels[unique_labels != 0])
    print(f"Final regions after merging: {num_final}")

    # Produce visualization
    color_vis = labelmap_to_color(labels)

    end = time.time()
    print(f"Time: {end - start:.4f}s")

    save_image(args.output, color_vis, is_binary=False)


else:
    print("Invalid command. Use --help.")