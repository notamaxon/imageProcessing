import argparse
import time
from PIL import Image
import numpy as np
import math
import cmath


parser = argparse.ArgumentParser(description="Frequency domain filtration tool (FFT Spatial Decimation & Filters)")


parser.add_argument('--command', type=str, required=True, 
                    help="Command to run: 'dft', 'fft', 'visualize', 'filter'")
parser.add_argument('--input', type=str, help="Input image file")
parser.add_argument('--output', type=str, help="Output image file")


parser.add_argument('--filter', type=str, 
                    choices=['low_pass', 'high_pass', 'band_pass', 'band_cut', 'edge_detect', 'phase_mod'],
                    help="Type of filter to apply (F1-F6)")


parser.add_argument('--radius', type=float, help="Radius for Low/High pass or edge detection filters")
parser.add_argument('--r_inner', type=float, help="Inner radius for Band pass/cut filters")
parser.add_argument('--r_outer', type=float, help="Outer radius for Band pass/cut filters")
parser.add_argument('--angle', type=float, help="Angle/Direction for Edge detection filter (F5) in degrees")
parser.add_argument('--param_k', type=int, default=0, help="Parameter k for Phase modifying filter (F6)")
parser.add_argument('--param_l', type=int, default=0, help="Parameter l for Phase modifying filter (F6)")

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
        image_array = np.abs(image_array)
        if is_binary:
            image_array = image_array * 255
        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        Image.fromarray(image_array).save(output_file)
        print(f"Image saved as {output_file}")
    except Exception as e:
        print(f"Error saving image: {e}")

def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()

def pad_image(image):
    h, w = image.shape
    new_h = next_power_of_2(h)
    new_w = next_power_of_2(w)
    
    if h == new_h and w == new_w:
        return image, h, w
        
    padded = np.zeros((new_h, new_w), dtype=image.dtype)
    padded[:h, :w] = image
    return padded, h, w

def crop_image(image, h, w):
    return image[:h, :w]

def dft_1d(vector):
    N = len(vector)
    X = np.zeros(N, dtype=complex)
    
    for k in range(N):
        current_sum = 0 + 0j
        
        for n in range(N):
            angle = -2j * np.pi * n * k / N
            W = np.exp(angle)
            
            current_sum += vector[n] * W
            
        X[k] = current_sum
        
    return X / math.sqrt(N)

def idft_1d(vector):
    N = len(vector)
    x = np.zeros(N, dtype=complex)
    
    for n in range(N):
        current_sum = 0 + 0j
        for k in range(N):
            angle = 2j * np.pi * n * k / N 
            W = np.exp(angle)
            
            current_sum += vector[k] * W
            
        x[n] = current_sum / math.sqrt(N)
        
    return x

def dft_2d(image):
    H, W = image.shape
    image_complex = image.astype(complex)
    
    row_transformed = np.zeros_like(image_complex)
    for r in range(H):
        row_transformed[r, :] = dft_1d(image_complex[r, :])
        
    final_transform = np.zeros_like(image_complex)
    for c in range(W):
        final_transform[:, c] = dft_1d(row_transformed[:, c])
        
    return final_transform

def idft_2d(spectrum):
    H, W = spectrum.shape
    row_restored = np.zeros_like(spectrum)
    for r in range(H):
        row_restored[r, :] = idft_1d(spectrum[r, :])
        
    final_image = np.zeros_like(spectrum)
    for c in range(W):
        final_image[:, c] = idft_1d(row_restored[:, c])
        
    return final_image

def fft_recursive(vector, inverse=False):
    N = len(vector)
    
    if N <= 1:
        return vector.astype(complex)
    
    even = fft_recursive(vector[0::2], inverse)
    odd  = fft_recursive(vector[1::2], inverse)
    
    k = np.arange(N // 2)
    sign = 1j if inverse else -1j
    W = np.exp(sign * 2 * np.pi * k / N)
    
    t = W * odd
    return np.concatenate([even + t, even - t])

def fft_1d_t1(vector):
    return fft_recursive(vector, inverse=False) / math.sqrt(len(vector))

def ifft_1d_t1(vector):
    result = fft_recursive(vector, inverse=True) / math.sqrt(len(vector))
    
    return result


def fft_2d(image):
    H, W = image.shape
    image_complex = image.astype(complex)
    
    row_fft = np.zeros_like(image_complex)
    for r in range(H):
        row_fft[r, :] = fft_1d_t1(image_complex[r, :])
        
    out = np.zeros_like(image_complex)
    for c in range(W):
        out[:, c] = fft_1d_t1(row_fft[:, c])
        
    return out

def ifft_2d(spectrum):
    H, W = spectrum.shape

    row_ifft = np.zeros_like(spectrum)
    for r in range(H):
        row_ifft[r, :] = ifft_1d_t1(spectrum[r, :])

    out = np.zeros_like(spectrum)
    for c in range(W):
        out[:, c] = ifft_1d_t1(row_ifft[:, c])
        
    return out


def fft_shift(spectrum):
    M, N = spectrum.shape
    mid_m, mid_n = M // 2, N // 2
    
    shifted = np.zeros_like(spectrum)
    
    shifted[mid_m:, mid_n:] = spectrum[0:mid_m, 0:mid_n]
    shifted[0:mid_m, 0:mid_n] = spectrum[mid_m:, mid_n:]
    shifted[mid_m:, 0:mid_n] = spectrum[0:mid_m, mid_n:]
    shifted[0:mid_m, mid_n:] = spectrum[mid_m:, 0:mid_n]
    
    return shifted


def generate_spectrum_visualization(spectrum):
    magnitude = np.abs(spectrum)

    spectrum_log = np.log(1 + magnitude)
    
    min_val = spectrum_log.min()
    max_val = spectrum_log.max()
    
    if max_val - min_val == 0:
        return np.zeros_like(spectrum_log, dtype=np.uint8)
        
    normalized = 255 * (spectrum_log - min_val) / (max_val - min_val)
    
    return normalized.astype(np.uint8)


def create_low_pass_mask(shape, radius):
    """
    (F1) Low-pass (high-cut) filter
    """
    H, W = shape
    u = np.arange(H) - H // 2
    v = np.arange(W) - W // 2
    U, V = np.meshgrid(u, v, indexing='ij')
    R = np.sqrt(U**2 + V**2)
    return (R <= radius).astype(float)


def create_high_pass_mask(shape, radius):
    """
    (F2) High-pass (low-cut) filter
    """
    return 1.0 - create_low_pass_mask(shape, radius)


def create_band_pass_mask(shape, r_inner, r_outer):
    """
    (F3) Band-pass filter
    """
    H, W = shape
    u = np.arange(H) - H // 2
    v = np.arange(W) - W // 2
    U, V = np.meshgrid(u, v, indexing='ij')
    R = np.sqrt(U**2 + V**2)
    return ((R >= r_inner) & (R <= r_outer)).astype(float)


def create_band_cut_mask(shape, r_inner, r_outer):
    """
    (F4) Band-cut (band-stop) filter
    """
    return 1.0 - create_band_pass_mask(shape, r_inner, r_outer)


def create_edge_detection_mask(shape, radius, angle_deg):
    """
    (F5) Directional high-pass filter
    """
    H, W = shape
    u = np.arange(H) - H // 2
    v = np.arange(W) - W // 2
    U, V = np.meshgrid(u, v, indexing='ij')

    R = np.sqrt(U**2 + V**2)
    angles = np.arctan2(V, U)

    angle_center = math.radians(angle_deg)
    angle_width = math.radians(15) 

    angle_diff = np.abs((angles - angle_center + np.pi) % (2*np.pi) - np.pi)

    return ((R > radius) & (angle_diff <= angle_width)).astype(float)


def create_phase_modifying_mask(shape, k, l):
    """
    (F6) Phase modifying filter
    """
    N, M = shape
    n = np.arange(N).reshape(N, 1)
    m = np.arange(M).reshape(1, M)

    phase = (
        -2 * np.pi * k * n / N
        -2 * np.pi * l * m / M
        + (k + l) * np.pi
    )

    return np.exp(1j * phase)





def apply_filter(input_file, output_file, filter_type):
    image = read_image(input_file)
    if image is None:
        return

    padded_img, h, w = pad_image(image)

    
    spectrum = fft_2d(padded_img)
    shifted = fft_shift(spectrum)
    
    shape = shifted.shape
    cy, cx = shape[0] // 2, shape[1] // 2
    dc_value = shifted[cy, cx]


    

    if filter_type == 'low_pass':
        mask = create_low_pass_mask(shape, args.radius)

    elif filter_type == 'high_pass':
        mask = create_high_pass_mask(shape, args.radius)

    elif filter_type == 'band_pass':
        mask = create_band_pass_mask(shape, args.r_inner, args.r_outer)

    elif filter_type == 'band_cut':
        mask = create_band_cut_mask(shape, args.r_inner, args.r_outer)

    elif filter_type == 'edge_detect':
        mask = create_edge_detection_mask(shape, args.radius, args.angle)

    elif filter_type == 'phase_mod':
        mask = create_phase_modifying_mask(shape, args.param_k, args.param_l)

    else:
        raise ValueError("Unknown filter type")

    
    filtered_shifted = shifted * mask

    if filter_type == 'high_pass':
        filtered_shifted[cy, cx] = dc_value

    
    unshifted = np.fft.ifftshift(filtered_shifted)
    restored = ifft_2d(unshifted)

    restored = crop_image(np.real(restored), h, w)

    save_image(output_file, restored)



if __name__ == "__main__":
    if args.command == 'dft':
        if args.input and args.output:
            img = read_image(args.input)
            if img is not None:
                start = time.time()
                freq_data = dft_2d(img)
                print(f"Forward DFT time: {time.time()-start:.4f}s")
                
                restored_complex = idft_2d(freq_data)
                save_image(args.output, restored_complex)

    elif args.command == 'fft':
            if args.input and args.output:
                img = read_image(args.input)
            print("Running Fast FFT (T1)...")
            padded_img, h, w = pad_image(img)
            print(f"Image padded from {h}x{w} to {padded_img.shape}")
            
            start = time.time()
            ft = fft_2d(padded_img)
            print(f"Forward FFT time: {time.time()-start:.4f}s")
            
            ift = ifft_2d(ft)
            
            restored = crop_image(ift, h, w)      
            save_image(args.output, restored)

    elif args.command == 'visualize':
        if args.input and args.output:
            image = read_image(args.input)
            if image is not None:
                print("Generating Fourier Spectrum Visualization...")
                padded_img, h, w = pad_image(image)
                spectrum = fft_2d(padded_img)
                shifted_spectrum = fft_shift(spectrum)
                vis_image = generate_spectrum_visualization(shifted_spectrum)
                save_image(args.output, vis_image)

    elif args.command == 'filter':
        if args.input and args.output and args.filter:
            
            
            params = {}
            if args.filter in ['low_pass', 'high_pass', 'edge_detect']:
                if args.radius is None:
                    print("Error: --radius is required for this filter.")
                    exit()
                params['radius'] = args.radius
                
            if args.filter in ['band_pass', 'band_cut']:
                if args.r_inner is None or args.r_outer is None:
                    print("Error: --r_inner and --r_outer are required for band filters.")
                    exit()
                params['r_inner'] = args.r_inner
                params['r_outer'] = args.r_outer

            if args.filter == 'edge_detect':
                if args.angle is None:
                    print("Error: --angle is required for edge detection filter.")
                    exit()
                params['angle'] = args.angle
            
            if args.filter == 'phase_mod':
                
                params['k'] = args.param_k
                params['l'] = args.param_l

            print(f"Applying filter: {args.filter} with params: {params}")
            apply_filter(args.input, args.output, args.filter)

    else:
        print("Invalid command. Use --help for usage information.")