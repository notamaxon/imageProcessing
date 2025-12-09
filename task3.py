import argparse
import time
from PIL import Image
import numpy as np
import sys

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
    se_h, se_w = se.shape
    pad_h, pad_w = se_h // 2, se_w // 2
    
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output = np.zeros_like(image)
    
    se_mask = (se == 1)
    
    for i in range(h):
        for j in range(w):
            roi = padded[i:i+se_h, j:j+se_w]
            if np.sum(roi[se_mask]) > 0:
                output[i, j] = 1
    return output


def erosion(image, se):
    h, w = image.shape
    se_h, se_w = se.shape
    pad_h, pad_w = se_h // 2, se_w // 2
    
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=1)
    output = np.zeros_like(image)
    
    se_mask = (se == 1)
    target_sum = np.sum(se_mask) 
    
    for i in range(h):
        for j in range(w):
            roi = padded[i:i+se_h, j:j+se_w]
            if np.sum(roi[se_mask]) == target_sum:
                output[i, j] = 1
    return output


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
    A = (image > 0).astype(np.uint8)

    B1_mask = (se == 1) 
    B2_mask = (se == 0) 

    er1 = erosion_mask(A, B1_mask)

    A_comp = 1 - A
    er2 = erosion_mask(A_comp, B2_mask)

    return (er1 & er2).astype(np.uint8)


def m6(image, get_structural_element_fn):
    mask_names = ['xii_1', 'xii_2', 'xii_3', 'xii_4', 'xii_5', 'xii_6', 'xii_7', 'xii_8']

    current = image.copy().astype(np.uint8)

    iteration = 0
    while True:
        iteration += 1
        prev = current.copy()

        for name in mask_names:
            base = get_structural_element_fn(name).copy()

            swapped = base.copy()
            tmp = swapped.copy()
            tmp[base == 1] = 0
            tmp[base == 0] = 1
            tmp[base == -1] = -1
            swapped = tmp

            hmt_res = hit_or_miss(current, swapped)

            current = (current | hmt_res).astype(np.uint8)

        if np.array_equal(current, prev):
            break

    return current


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
    image = read_image(args.input, as_binary=False)
    if image is None: sys.exit(1)

    start = time.time()
    print("Region growing not implemented yet")
    output_image = image 
    end = time.time()
    save_image(args.output, output_image, is_binary=False)

else:
    print("Invalid command.")