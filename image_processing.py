import argparse
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description="Image reading and writing tool using NumPy")
parser.add_argument('--command', type=str, help="Command to run (e.g., readwrite, help)")
parser.add_argument('--input', type=str, help="Input BMP image file")
parser.add_argument('--output', type=str, help="Output BMP image file")
parser.add_argument('--brightness', type=int, help="Brightness adjustment value (positive to increase, negative to decrease)")
parser.add_argument('--contrast', type=int, help="Contrast adjustment value (negative to decrease, positive to increase)")
parser.add_argument('--negative', action='store_true', help="Apply negative effect")
parser.add_argument('--hflip', action='store_true', help="Apply horizontal flip")
parser.add_argument('--vflip', action='store_true', help="Apply vertical flip")
parser.add_argument('--dflip', action='store_true', help="Apply diagonal flip")
parser.add_argument('--shrink', type=int, help="Scale factor for shrinking the image")
parser.add_argument('--enlarge', type=int, help="Scale factor for enlarging the image")
parser.add_argument('--alpha', type=int, help="Alpha parameter for Alpha-trimmed mean filter (must be even)")
parser.add_argument('--mse', action='store_true', help="Calculate Mean Square Error")
parser.add_argument('--pmse', action='store_true', help="Calculate Peak Mean Square Error")
parser.add_argument('--snr', action='store_true', help="Calculate Signal to Noise Ratio")
parser.add_argument('--psnr', action='store_true', help="Calculate Peak Signal to Noise Ratio")
parser.add_argument('--md', action='store_true', help="Calculate Maximum Difference")
parser.add_argument('--eval_all', action='store_true', help="Evaluate all metrics between input and output images")
parser.add_argument('--q', type=float, help="Q parameter for Contraharmonic mean filter")
parser.add_argument('--wsize', type=int, help="Window size parameter for filters")
args = parser.parse_args()


def read_and_write_image(input_file, output_file):
    try:
        im = Image.open(input_file)

        arr = np.array(im)
        new_im = Image.fromarray(arr.astype(np.uint8))

        new_im.save(output_file)
        print(f"Image read from {input_file} and saved as {output_file}")

    except Exception as e:
        print(f"Error occured while handling image: {e}")

def adjust_brightness(input_file, output_file, brightness_value):
    try:
        
        im = Image.open(input_file)

        arr = np.array(im)

        arr = arr.astype(np.int32)  # Convert to int32 to prevent overflow
        arr += brightness_value  
        arr = np.clip(arr, 0, 255)  
 
        new_im = Image.fromarray(arr.astype(np.uint8))

        new_im.save(output_file)
        print(f"Brightness adjusted image saved as {output_file}")

    except Exception as e:
        print(f"Error occurred while adjusting brightness: {e}")

def adjust_contrast(input_file, output_file, contrast_value):
    try:
        im = Image.open(input_file)

        arr = np.array(im)

        factor = (259 * (contrast_value + 255)) / (255 * (259 - contrast_value))

        arr = arr.astype(np.int32)
        arr = factor * (arr - 128) + 128
        arr = np.clip(arr, 0, 255)

        new_im = Image.fromarray(arr.astype(np.uint8))

        new_im.save(output_file)
        print(f"Contrast adjusted image saved as {output_file}")

    except Exception as e:
        print(f"Error occured while adjusting contrast: {e}")    

    
def apply_negative(input_file, output_file):
    try:
        im = Image.open(input_file)
        arr = np.array(im)

        
        arr = 255 - arr  
        
        new_im = Image.fromarray(arr.astype(np.uint8))
        new_im.save(output_file)
        print(f"Negative image saved as {output_file}")
    except Exception as e:
        print(f"Error occurred while applying negative: {e}")


def horizontal_flip(input_file, output_file):
    try:
        im = Image.open(input_file)

        arr = np.array(im)

        height = arr.shape[0]
        width = arr.shape[1]
        flipped_arr = np.zeros_like(arr)  
            
        for y in range(height):
            for x in range(width):
                flipped_arr[y, width - 1 - x] = arr[y, x]
        
        
        new_im = Image.fromarray(flipped_arr.astype(np.uint8))

        new_im.save(output_file)
        print(f"Horizontal flipped image saved as {output_file}")

    except Exception as e:
        print(f"Error occurred while performing horizontal flip: {e}")

def vertical_flip(input_file, output_file):
    try:
        im = Image.open(input_file)

        arr = np.array(im)

        
        height = arr.shape[0]
        width = arr.shape[1]
        flipped_arr = np.zeros_like(arr)
        for y in range(height):
            flipped_arr[height - 1 - y, :] = arr[y, :]

        new_im = Image.fromarray(flipped_arr.astype(np.uint8))

        new_im.save(output_file)
        print(f"Vertical flipped image saved as {output_file}")

    except Exception as e:
        print(f"Error occurred while performing vertical flip: {e}")

def diagonal_flip(input_file, output_file):
    try:
        im = Image.open(input_file)
        arr = np.array(im)
          
        height = arr.shape[0]
        width = arr.shape[1]
        flipped_arr = np.zeros_like(arr)
    
        for y in range(height):
            for x in range(width):
                flipped_arr[height - 1 - y, width - 1 - x] = arr[y, x]

        new_im = Image.fromarray(flipped_arr.astype(np.uint8))
        new_im.save(output_file)
        print(f"Diagonal (mirrored) flipped image saved as {output_file}")

    except Exception as e:
        print(f"Error occurred while performing diagonal flip: {e}")



def shrink(input_file, output_file, scale_factor):
    try:
        im = Image.open(input_file)
        arr = np.array(im)

        new_height = arr.shape[0] // scale_factor
        new_width = arr.shape[1] // scale_factor

        if arr.ndim == 2:  
            shrunk_arr = np.zeros((new_height, new_width), dtype=arr.dtype)
        else:  
            shrunk_arr = np.zeros((new_height, new_width, arr.shape[2]), dtype=arr.dtype)

        for y in range(new_height):
            for x in range(new_width):
                shrunk_arr[y, x] = arr[y * scale_factor, x * scale_factor]

        new_im = Image.fromarray(shrunk_arr.astype(np.uint8))
        new_im.save(output_file)
        print(f"Shrunken image saved as {output_file}")

    except Exception as e:
        print(f"Error occurred while shrinking the image: {e}")


def enlarge_image(input_file, output_file, scale_factor):
    try:
        im = Image.open(input_file)
        arr = np.array(im)

        height, width = arr.shape[:2]
        
        new_height = height * scale_factor
        new_width = width * scale_factor
        
        if arr.ndim == 2:  
            enlarged_arr = np.zeros((new_height, new_width), dtype=arr.dtype)
        else:  
            enlarged_arr = np.zeros((new_height, new_width, arr.shape[2]), dtype=arr.dtype)

        # bilinear interpolation
        for y in range(new_height):
            for x in range(new_width):

                src_y = y / scale_factor
                src_x = x / scale_factor
                
                x0 = int(src_x)
                y0 = int(src_y)
                
                x_diff = src_x - x0
                y_diff = src_y - y0
                
                # Get the neighboring pixels
                if x0 + 1 < width and y0 + 1 < height:
                    top_left = arr[y0, x0]
                    top_right = arr[y0, x0 + 1]
                    bottom_left = arr[y0 + 1, x0]
                    bottom_right = arr[y0 + 1, x0 + 1]

                    top = (1 - x_diff) * top_left + x_diff * top_right
                    bottom = (1 - x_diff) * bottom_left + x_diff * bottom_right
                    enlarged_arr[y, x] = (1 - y_diff) * top + y_diff * bottom
        
        new_im = Image.fromarray(enlarged_arr.astype(np.uint8))
        new_im.save(output_file)
        print(f"Enlarged image saved as {output_file}")

    except Exception as e:
        print(f"Error occurred while enlarging image: {e}")


def alpha_trimmed_mean_filter(image, window_size=5, alpha=2):
    padding = window_size // 2
    if image.ndim == 2: 
        padded_image = np.pad(image, padding, mode='constant', constant_values=0)
        output_image = np.zeros_like(image)
        
        for x in range(padding, padded_image.shape[0] - padding):
            for y in range(padding, padded_image.shape[1] - padding):
                window = padded_image[x - padding:x + padding + 1, y - padding:y + padding + 1]
                trimmed_window = np.sort(window.flatten())[alpha//2 : -alpha//2]
                output_image[x - padding, y - padding] = np.mean(trimmed_window)
                
    else:  
        output_image = np.zeros_like(image)
        for c in range(image.shape[2]):  
            padded_channel = np.pad(image[:, :, c], padding, mode='constant', constant_values=0)
            
            for x in range(padding, padded_channel.shape[0] - padding):
                for y in range(padding, padded_channel.shape[1] - padding):
                    window = padded_channel[x - padding:x + padding + 1, y - padding:y + padding + 1]
                    trimmed_window = np.sort(window.flatten())[alpha//2 : -alpha//2]
                    output_image[x - padding, y - padding, c] = np.mean(trimmed_window)
                    
    return output_image

def contraharmonic_mean_filter(image, window_size=3, Q=1.5):
    padding = window_size // 2
    if image.ndim == 2:  
        padded_image = np.pad(image, padding, mode='constant', constant_values=0)
        output_image = np.zeros_like(image, dtype=float)

        for x in range(padding, padded_image.shape[0] - padding):
            for y in range(padding, padded_image.shape[1] - padding):
                window = padded_image[x - padding:x + padding + 1, y - padding:y + padding + 1]
                
                non_zero_window = window[window != 0]
                
                if non_zero_window.size > 0:  
                    numerator = np.sum(non_zero_window ** (Q + 1))
                    denominator = np.sum(non_zero_window ** Q)
                    output_image[x - padding, y - padding] = numerator / denominator if denominator != 0 else 0
                else:
                    output_image[x - padding, y - padding] = 0  

    else: 
        output_image = np.zeros_like(image, dtype=float)
        for c in range(image.shape[2]):  
            padded_channel = np.pad(image[:, :, c], padding, mode='constant', constant_values=0)
            for x in range(padding, padded_channel.shape[0] - padding):
                for y in range(padding, padded_channel.shape[1] - padding):
                    window = padded_channel[x - padding:x + padding + 1, y - padding:y + padding + 1]
                    
                    non_zero_window = window[window != 0]
                    
                    if non_zero_window.size > 0:
                        numerator = np.sum(non_zero_window ** (Q + 1))
                        denominator = np.sum(non_zero_window ** Q)
                        output_image[x - padding, y - padding, c] = numerator / denominator if denominator != 0 else 0
                    else:
                        output_image[x - padding, y - padding, c] = 0  
                    
    return np.clip(output_image, 0, 255)

# Function to apply Alpha-trimmed mean filter and save result
def apply_alpha_trimmed(input_file, output_file, alpha, wsize):
    im = Image.open(input_file)
    arr = np.array(im)  
    filtered_image = alpha_trimmed_mean_filter(arr, alpha=alpha, window_size=wsize)
    new_im = Image.fromarray(filtered_image.astype(np.uint8))
    new_im.save(output_file)
    print(f"Alpha-trimmed mean filtered image saved as {output_file}")

# Function to apply Contraharmonic Mean Filter and save result
def apply_contraharmonic(input_file, output_file, Q, wsize):
    im = Image.open(input_file)
    arr = np.array(im)
    filtered_image = contraharmonic_mean_filter(arr, Q=Q, window_size=wsize)
    new_im = Image.fromarray(filtered_image.astype(np.uint8))
    new_im.save(output_file)
    print(f"Contraharmonic mean filtered image saved as {output_file}")

def mean_square_error(original, processed):
    return np.mean((original - processed) ** 2, axis=(0, 1)).mean()

def peak_mean_square_error(original, processed):
    max_vals = np.max(original, axis=(0, 1))
    mse = mean_square_error(original, processed)
    return np.mean(mse / (max_vals ** 2)) if np.any(max_vals != 0) else float('inf')

def signal_to_noise_ratio(original, processed):
    signal_power = np.sum(original ** 2, axis=(0, 1))
    noise_power = np.sum((original - processed) ** 2, axis=(0, 1))
    return np.mean(10 * np.log10(signal_power / noise_power)) if np.all(noise_power != 0) else float('inf')

def peak_signal_to_noise_ratio(original, processed):
    mse_per_channel = np.mean((original - processed) ** 2, axis=(0, 1))
    max_pixel_value = 255
    psnr_per_channel = 10 * np.log10((max_pixel_value ** 2) / mse_per_channel)
    return np.mean(psnr_per_channel) if np.all(mse_per_channel != 0) else float('inf')

def maximum_difference(original, processed):
    return np.max(np.abs(original - processed), axis=(0, 1))

# Function to evaluate all metrics at once
def evaluate_all_metrics(original_path, processed_path):
    original = np.array(Image.open(original_path))
    processed = np.array(Image.open(processed_path))
    
    mse = mean_square_error(original, processed)
    pmse = peak_mean_square_error(original, processed)
    snr = signal_to_noise_ratio(original, processed)
    psnr = peak_signal_to_noise_ratio(original, processed)
    md = maximum_difference(original, processed)
    
    print(f"Mean Square Error (MSE): {mse}")
    print(f"Peak Mean Square Error (PMSE): {pmse}")
    print(f"Signal to Noise Ratio (SNR): {snr} dB")
    print(f"Peak Signal to Noise Ratio (PSNR): {psnr} dB")
    print(f"Maximum Difference (MD): {md}")


if args.command == 'readwrite':
    if args.input and args.output:
        read_and_write_image(args.input, args.output)
    else:
        print("Please provide both input and output image files")
        
elif args.command == 'brightness':
    if args.input and args.output and args.brightness is not None:
        adjust_brightness(args.input, args.output, args.brightness)
    else:
        print("Please provide input, output image files, and brightness value")

elif args.command == 'contrast':
    if args.input and args.output and args.contrast is not None:
        adjust_contrast(args.input, args.output, args.contrast)
    else:
        print("Please provide input and ouput image files, and a contrast value")    

elif args.command == 'negative':
    if args.input and args.output:
        apply_negative(args.input, args.output)
    else:
        print("Please provide input and output image files")

elif args.command == 'hflip':
    if args.input and args.output:
        horizontal_flip(args.input, args.output)
    else:
        print("Please provide input and output image files") 

elif args.command == 'vflip':
    if args.input and args.output:
        vertical_flip(args.input, args.output)
    else:
        print("Please provide input and output image files")

elif args.command == 'dflip':
    if args.input and args.output:
        diagonal_flip(args.input, args.output)
    else:
        print("Please provide input and output image files")

elif args.command == 'shrink':
    if args.input and args.output and args.shrink is not None:
        shrink(args.input, args.output, args.shrink)
    else:
        print("Please provide input, output image files, and a scale factor")

elif args.command == 'enlarge':
    if args.input and args.output and args.enlarge is not None:
        enlarge_image(args.input, args.output, args.enlarge)
    else:
        print("Please provide input, output image files, and a scale factor for enlargement")

elif args.command == 'alpha':
    if args.input and args.output and args.alpha and args.wsize is not None:
        apply_alpha_trimmed(args.input, args.output, args.alpha, args.wsize)
    else:
        print("Please provide input and output image files, alpha value and window size")

elif args.command == 'cmean':
    if args.input and args.output and args.q and args.wsize is not None:
        apply_contraharmonic(args.input, args.output, args.q, args.wsize)
    else:
        print("Please provide input and output image files, q value and window size")

elif args.command == "mse":
    if args.input and args.output:
        original = np.array(Image.open(args.input))
        processed = np.array(Image.open(args.output))
        print(f"Mean Square Error (MSE): {mean_square_error(original, processed)}")
    else:
        print("Please provide both input and output image files for MSE calculation.")

elif args.command == "pmse":
    if args.input and args.output:
        original = np.array(Image.open(args.input))
        processed = np.array(Image.open(args.output))
        print(f"Peak Mean Square Error (PMSE): {peak_mean_square_error(original, processed)}")
    else:
        print("Please provide both input and output image files for PMSE calculation.")

elif args.command == "snr":
    if args.input and args.output:
        original = np.array(Image.open(args.input))
        processed = np.array(Image.open(args.output))
        print(f"Signal to Noise Ratio (SNR): {signal_to_noise_ratio(original, processed)} dB")
    else:
        print("Please provide both input and output image files for SNR calculation.")

elif args.command == "psnr":
    if args.input and args.output:
        original = np.array(Image.open(args.input))
        processed = np.array(Image.open(args.output))
        print(f"Peak Signal to Noise Ratio (PSNR): {peak_signal_to_noise_ratio(original, processed)} dB")
    else:
        print("Please provide both input and output image files for PSNR calculation.")

elif args.command == "md":
    if args.input and args.output:
        original = np.array(Image.open(args.input))
        processed = np.array(Image.open(args.output))
        print(f"Maximum Difference (MD): {maximum_difference(original, processed)}")
    else:
        print("Please provide both input and output image files for MD calculation.")

elif args.command == "eval_all":
    if args.input and args.output:
        evaluate_all_metrics(args.input, args.output)
    else:
        print("Please provide both input and output image files for evaluating all metrics.")
        
elif args.command == 'help':
    parser.print_help()
else:
    print("Invalid command. Use --help for available commands.")


