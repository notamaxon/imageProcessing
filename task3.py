import argparse
import time
from PIL import Image
import numpy as np
import sys

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
    # Load as Grayscale (NOT binary)
    image = read_image(args.input, as_binary=False)
    if image is None: sys.exit(1)

    start = time.time()
    
    # Placeholder for R1
    print("Region growing not implemented yet")
    output_image = image # Just pass through for now
    
    end = time.time()
    print(f"Time: {end - start:.4f}s")
    save_image(args.output, output_image, is_binary=False)

else:
    print("Invalid command. Use --help.")