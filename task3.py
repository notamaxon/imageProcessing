import argparse
import time
from PIL import Image
import numpy as np
import sys
from collections import deque, defaultdict
import math
import random

parser = argparse.ArgumentParser(description="Image processing tool: Morphology and Segmentation")

parser.add_argument('--command', type=str, required=True, 
                    help="Command to run: dilation, erosion, opening, closing, hmt, m6, region_growing")
parser.add_argument('--input', type=str, required=True, help="Input BMP image file")
parser.add_argument('--output', type=str, required=True, help="Output BMP image file")

parser.add_argument('--se', type=str, default='iii', 
                    help="Structural element identifier (e.g., 'iii', 'xi_1', 'xii_3')")

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
parser.add_argument(
    '--criterion', type=str, default='intensity',
    choices=['intensity', 'color', 'texture'],
    help="Homogeneity criterion for region growing (intensity, color, texture)"
)


args = parser.parse_args()


def read_image(input_file, as_binary=False):
    try:
        img = Image.open(input_file).convert('L')
        arr = np.array(img)
        if as_binary:
            arr = (arr > 127).astype(np.uint8)
        return arr
    except Exception as e:
        print(f"Error reading image: {e}")
        return None

def save_image(output_file, image_array, is_binary=False):
    try:
        if is_binary:
            image_array = image_array * 255
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        Image.fromarray(image_array).save(output_file)
        print(f"Image saved as {output_file}")
    except Exception as e:
        print(f"Error saving image: {e}")


def get_structural_element(name):
    masks = {
        'i': np.array([
            [ 0, 0, 0],
            [ 0,  1, 1],
            [ 0, 0, 0]
        ], dtype=np.int8),
        'ii': np.array([
            [ 0, 0, 0],
            [ 0,  1, 0],
            [ 0, 1, 0]
        ], dtype=np.int8),
        'iii': np.array([
            [ 1, 1, 1],
            [ 1, 1, 1],
            [ 1, 1, 1]
        ], dtype=np.int8),
        'iv': np.array([
            [ 0, 1, 0],
            [ 1,  1, 1],
            [ 0, 1, 0]
        ], dtype=np.int8),
        'v': np.array([
            [ 0, 0, 0],
            [ 0,  1, 1],
            [ 0, 1, 0]
        ], dtype=np.int8),
        'vi': np.array([
            [ 0, 0, 0],
            [ 0,  0, 1],
            [ 0, 1, 0]
        ], dtype=np.int8),
        'vii': np.array([
            [ 0, 0, 0],
            [ 1, 1, 1],
            [ 0, 0, 0]
        ], dtype=np.int8),
        'viii': np.array([
            [ 0, 0, 0],
            [ 1, 0, 1],
            [ 0, 0, 0]
        ], dtype=np.int8),
        'ix': np.array([
            [ 0, 0, 0],
            [ 1, 1, 0],
            [ 1, 0, 0]
        ], dtype=np.int8),
        'x': np.array([
            [ 0, 1, 1],
            [ 0, 1, 0],
            [ 0, 0, 0]
        ], dtype=np.int8),

        'xi_1': np.array([
            [ 1, -1, -1],
            [ 1,  0, -1],
            [ 1, -1, -1]
        ], dtype=np.int8),
        'xi_2': np.array([
            [ 1,  1,  1],
            [ -1,  0, -1],
            [ -1,  -1, -1]
        ], dtype=np.int8),
        'xi_3': np.array([
            [ -1,  -1,  1],
            [ -1,  0,  1],
            [-1, -1, 1]
        ], dtype=np.int8),
        'xi_4': np.array([
            [-1,  -1,  -1],
            [-1,  0,  -1],
            [1,  1,  1]
        ], dtype=np.int8),

        'xii_1': np.array([
            [ 0,  0,  0],
            [-1, 1, -1],
            [ 1,  1,  1]
        ], dtype=np.int8),
        'xii_2': np.array([
            [-1,  0,  0],
            [ 1, 1,  0],
            [ 1,  1, -1]
        ], dtype=np.int8),
        'xii_3': np.array([
            [ 1, -1,  0],
            [ 1, 1,  0],
            [ 1, -1,  0]
        ], dtype=np.int8),
        'xii_4': np.array([
            [ 1,  1, -1],
            [ 1, 1,  0],
            [-1,  0,  0]
        ], dtype=np.int8),
        'xii_5': np.array([
            [ 1,  1,  1],
            [-1, 1, -1],
            [ 0,  0,  0]
        ], dtype=np.int8),
        'xii_6': np.array([
            [-1,  1,  1],
            [ 0, 1,  1],
            [ 0,  0, -1]
        ], dtype=np.int8),
        'xii_7': np.array([
            [ 0, -1,  1],
            [ 0, 1,  1],
            [ 0, -1,  1]
        ], dtype=np.int8),
        'xii_8': np.array([
            [ 0,  0, -1],
            [ 0, 1,  1],
            [-1,  1,  1]
        ], dtype=np.int8)
    }

    if name not in masks:
        print(f"Error: Structural element '{name}' not defined.")
        sys.exit(1)
        
    return masks[name]



def dilation(image, se):
    h, w = image.shape
    sh, sw = se.shape
    ph, pw = sh//2, sw//2
    padded = np.pad(image, ((ph,ph),(pw,pw)), mode='constant', constant_values=0)
    out = np.zeros_like(image)
    mask = (se==1)
    for i in range(h):
        for j in range(w):
            roi = padded[i:i+sh, j:j+sw]
            if np.sum(roi[mask])>0:
                out[i,j]=1
    return out

def erosion(image, se):
    h, w = image.shape
    sh, sw = se.shape
    ph, pw = sh//2, sw//2
    padded = np.pad(image, ((ph,ph),(pw,pw)), mode='constant', constant_values=1)
    out = np.zeros_like(image)
    mask = (se==1)
    req = np.sum(mask)
    for i in range(h):
        for j in range(w):
            roi = padded[i:i+sh, j:j+sw]
            if np.sum(roi[mask])==req:
                out[i,j]=1
    return out


def opening(image, se):
    return dilation(erosion(image, se), se)


def closing(image, se):
    return erosion(dilation(image, se), se)


def erosion_mask(image, mask):
    h, w = image.shape
    se_h, se_w = mask.shape
    pad_h, pad_w = se_h // 2, se_w // 2

    if np.sum(mask) == 0:
        return image.copy()

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)),
                    mode='constant', constant_values=1)

    out = np.zeros_like(image, dtype=np.uint8)
    required = np.sum(mask)

    for i in range(h):
        for j in range(w):
            roi = padded[i:i+se_h, j:j+se_w]
            if roi[mask].sum() == required:
                out[i, j] = 1
    return out


def hit_or_miss(image, se):
    A=(image>0).astype(np.uint8)
    B1=(se==1)
    B2=(se==0)
    er1=erosion_mask(A,B1)
    Acomp=1-A
    er2=erosion_mask(Acomp,B2)
    return (er1 & er2).astype(np.uint8)

def m6(image, get_se):
    names=['xii_1','xii_2','xii_3','xii_4','xii_5','xii_6','xii_7','xii_8']
    current=image.copy().astype(np.uint8)
    while True:
        prev=current.copy()
        for name in names:
            base=get_se(name)
            tmp=base.copy()
            tmp[base==1]=0
            tmp[base==0]=1
            tmp[base==-1]=-1
            res=hit_or_miss(current,tmp)
            current=(current|res).astype(np.uint8)
        if np.array_equal(current,prev):
            break
    return current

    
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

# ---------------- TEXTURE SUPPORT: LOCAL STANDARD DEVIATION ----------------

def integral_image(img):
    """Compute (H+1)x(W+1) summed-area table for fast window sums."""
    img = img.astype(np.float64)
    H, W = img.shape
    ii = np.zeros((H+1, W+1), dtype=np.float64)
    ii[1:, 1:] = img.cumsum(axis=0).cumsum(axis=1)
    return ii

def local_sum(ii, x0, y0, x1, y1):
    """Sum of rectangle [x0..x1-1, y0..y1-1] using integral image."""
    return ii[x1, y1] - ii[x0, y1] - ii[x1, y0] + ii[x0, y0]

def compute_local_std_map(gray, k=5):
    """
    Compute local standard deviation per pixel using integral images.
    Window size = k (must be odd). Returned array is float64 (H,W).
    """
    assert k % 2 == 1
    H, W = gray.shape
    pad = k // 2

    # Reflect padding
    padded = np.pad(gray.astype(np.float64), pad, mode='reflect')

    ii = integral_image(padded)
    ii2 = integral_image(padded * padded)

    std_map = np.zeros((H, W), dtype=np.float64)
    area = k * k

    for i in range(H):
        for j in range(W):
            x0, y0 = i, j
            x1, y1 = x0 + k, y0 + k
            s = local_sum(ii,  x0, y0, x1, y1)
            s2 = local_sum(ii2, x0, y0, x1, y1)
            mean = s / area
            var = (s2 / area) - (mean * mean)
            std_map[i, j] = math.sqrt(max(var, 0.0))

    return std_map



def region_growing_dynamic(image, seeds, threshold, adjacency=8,
                           criterion='intensity', descriptor_map=None):
    """
    Region growing that updates region mean on the fly.
    - image: ndarray (H,W) grayscale or (H,W,3) color
    - seeds: list of (row, col)
    Returns: labels (H,W int32), num_regions, region_stats
    """
    if criterion == 'texture' and descriptor_map is None:
        raise ValueError("descriptor_map must be provided for texture criterion")
    
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
        if criterion == 'texture':
            # descriptor_map is a float64 std_map computed before calling this function
            tex_val = float(descriptor_map[sx, sy])
            region_stats[current_label]['tex_sum'] = tex_val
            region_stats[current_label]['tex_mean'] = tex_val
        labels[sx, sy] = current_label

        q = deque()
        q.append((sx, sy))

        while q:
            x, y = q.popleft()
            for nx, ny in get_neighbors(x, y, h, w, adjacency):
                if labels[nx, ny] != 0:
                    continue
                # distance between neighbor pixel and current region mean
                # ---------------- HOMOGENEITY CRITERIA ----------------
                if criterion == 'intensity':
                    # grayscale expected; difference vs region mean
                    dist = abs(int(image[nx, ny]) - region_stats[current_label]['mean'])

                elif criterion == 'color':
                    # RGB Euclidean distance
                    mean = region_stats[current_label]['mean']
                    px = image[nx, ny]
                    dist = math.sqrt(sum((mean[i] - int(px[i]))**2 for i in range(3)))

                elif criterion == 'texture':
                    # descriptor_map must be the std_map
                    pixel_std = descriptor_map[nx, ny]
                    region_std_mean = region_stats[current_label]['tex_mean']
                    dist = abs(pixel_std - region_std_mean)


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
                    if criterion == 'texture':
                        tex_val = float(descriptor_map[nx, ny])
                        region_stats[current_label]['tex_sum'] += tex_val
                        region_stats[current_label]['tex_mean'] = region_stats[current_label]['tex_sum'] / region_stats[current_label]['count']
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

morphology_cmds = ['dilation', 'erosion', 'opening', 'closing', 'hmt', 'm6']

if args.command in morphology_cmds:
    image = read_image(args.input, as_binary=True)
    if image is None: sys.exit(1)
    
    start = time.time()
    output_image = None
    
    if args.command == 'dilation':
        se = get_structural_element(args.se)
        output_image = dilation(image, se)

    elif args.command == 'erosion':
        se = get_structural_element(args.se)
        output_image = erosion(image, se)

    elif args.command == 'opening':
        se = get_structural_element(args.se)
        output_image = opening(image, se)

    elif args.command == 'closing':
        se = get_structural_element(args.se)
        output_image = closing(image, se)

    elif args.command == 'hmt':
        se = get_structural_element(args.se)
        output_image = hit_or_miss(image, se)
        
    elif args.command == 'm6':
        print("Running M6: Thickening with SE (xii)...")
        output_image = m6(image, get_structural_element)

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

    print(f"Using {len(seeds)} seed(s). Adjacency={args.adjacency}, threshold={args.threshold}, criterion={args.criterion}")
    if args.criterion == 'texture':
        # build grayscale version for descriptor computation
        if img_arr.ndim == 3:
            gray = np.mean(img_arr, axis=2).astype(np.uint8)
        else:
            gray = img_arr
        print("Computing texture descriptor (local std, k=5)...")
        descriptor_map = compute_local_std_map(gray, k=5)
    else:
        descriptor_map = None

    labels, num_regions, stats = region_growing_dynamic(
        img_arr, seeds, args.threshold, adjacency=args.adjacency,
        criterion=args.criterion, descriptor_map=descriptor_map
    )
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
    print("Invalid command.")