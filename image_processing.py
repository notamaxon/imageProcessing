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

        if arr.ndim == 2:  # Grayscale image
            arr = arr.astype(np.int32)  # Convert to int32 to prevent overflow
            arr += brightness_value  
            arr = np.clip(arr, 0, 255)  
        else:  # RGB or other multi-channel image
            arr = arr.astype(np.int32)  
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

        if arr.ndim == 2:  
            height, width = arr.shape
            flipped_arr = np.zeros_like(arr)  
            
            for y in range(height):
                for x in range(width):
                    flipped_arr[y, width - 1 - x] = arr[y, x]
        else:  
            height, width, channels = arr.shape
            flipped_arr = np.zeros_like(arr) 
            
            for y in range(height):
                for x in range(width):
                    flipped_arr[y, width - 1 - x, :] = arr[y, x, :]  
        
        new_im = Image.fromarray(flipped_arr.astype(np.uint8))

        new_im.save(output_file)
        print(f"Horizontal flipped image saved as {output_file}")

    except Exception as e:
        print(f"Error occurred while performing horizontal flip: {e}")

def vertical_flip(input_file, output_file):
    try:
        im = Image.open(input_file)

        arr = np.array(im)

        if arr.ndim == 2:
            height, width = arr.shape
            flipped_arr = np.zeros_like(arr)
            for y in range(height):
                flipped_arr[height - 1 - y, :] = arr[y, :]

        else:  
            height, width, channels = arr.shape
            flipped_arr = np.zeros_like(arr)
            for y in range(height):
                flipped_arr[height - 1 - y, :, :] = arr[y, :, :]
        
        new_im = Image.fromarray(flipped_arr.astype(np.uint8))

        new_im.save(output_file)
        print(f"Vertical flipped image saved as {output_file}")

    except Exception as e:
        print(f"Error occurred while performing vertical flip: {e}")

def diagonal_flip(input_file, output_file):
    try:
        im = Image.open(input_file)
        arr = np.array(im)

        if arr.ndim == 2:  
            height, width = arr.shape
            flipped_arr = np.zeros((height, width), dtype=arr.dtype)
            
            for y in range(height):
                for x in range(width):
                    flipped_arr[height - 1 - y, width - 1 - x] = arr[y, x]

        else:  
            height, width, channels = arr.shape
            flipped_arr = np.zeros((height, width, channels), dtype=arr.dtype)
            
            for y in range(height):
                for x in range(width):
                    flipped_arr[height - 1 - y, width - 1 - x, :] = arr[y, x, :]

        new_im = Image.fromarray(flipped_arr.astype(np.uint8))
        new_im.save(output_file)
        print(f"Diagonal (mirrored) flipped image saved as {output_file}")

    except Exception as e:
        print(f"Error occurred while performing diagonal flip: {e}")



def shrink(input_file, output_file, scale_factor):
    try:
        im = Image.open(input_file)
        arr = np.array(im)

        # Calculate the new dimensions
        new_height = arr.shape[0] // scale_factor
        new_width = arr.shape[1] // scale_factor

        # Create an empty array for the shrunken image
        if arr.ndim == 2:  # Grayscale image
            shrunk_arr = np.zeros((new_height, new_width), dtype=arr.dtype)
        else:  # RGB or multi-channel image
            shrunk_arr = np.zeros((new_height, new_width, arr.shape[2]), dtype=arr.dtype)

        # Fill the shrunk array
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
        
        enlarged_arr = np.zeros((new_height, new_width, arr.shape[2]), dtype=arr.dtype) if arr.ndim == 3 else np.zeros((new_height, new_width), dtype=arr.dtype)

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


def alpha_trimmed_mean_filter(image, window_size=3, alpha=2):
    padding = window_size // 2
    if image.ndim == 2:  # Grayscale image
        padded_image = np.pad(image, padding, mode='constant', constant_values=0)
        output_image = np.zeros_like(image)
        
        for x in range(padding, padded_image.shape[0] - padding):
            for y in range(padding, padded_image.shape[1] - padding):
                # Extract and process the window
                window = padded_image[x - padding:x + padding + 1, y - padding:y + padding + 1]
                trimmed_window = np.sort(window.flatten())[alpha//2 : -alpha//2]
                output_image[x - padding, y - padding] = np.mean(trimmed_window)
                
    else:  # Color image
        output_image = np.zeros_like(image)
        for c in range(image.shape[2]):  # Loop over color channels
            padded_channel = np.pad(image[:, :, c], padding, mode='constant', constant_values=0)
            
            for x in range(padding, padded_channel.shape[0] - padding):
                for y in range(padding, padded_channel.shape[1] - padding):
                    # Extract and process the window for the current channel
                    window = padded_channel[x - padding:x + padding + 1, y - padding:y + padding + 1]
                    trimmed_window = np.sort(window.flatten())[alpha//2 : -alpha//2]
                    output_image[x - padding, y - padding, c] = np.mean(trimmed_window)
                    
    return output_image

# Function to apply Alpha-trimmed mean filter and save result
def apply_alpha_trimmed(input_file, output_file, alpha):
    im = Image.open(input_file)
    arr = np.array(im)  # Convert to grayscale
    filtered_image = alpha_trimmed_mean_filter(arr, alpha=alpha)
    new_im = Image.fromarray(filtered_image.astype(np.uint8))
    new_im.save(output_file)
    print(f"Alpha-trimmed mean filtered image saved as {output_file}")


def calculate_similarity_measures(original_file, processed_file):
    try:
        original = Image.open(original_file)
        processed = Image.open(processed_file)

        original_arr = np.array(original)
        processed_arr = np.array(processed)

        # Ensure arrays are the same shape
        if original_arr.shape != processed_arr.shape:
            print("Original and processed images must have the same dimensions for similarity measures.")
            return

        mse = np.mean((original_arr - processed_arr) ** 2)
        pmse = np.max(original_arr) ** 2 / mse if mse != 0 else 0
        snr = 10 * np.log10(np.sum(original_arr ** 2) / (np.sum((original_arr - processed_arr) ** 2) + 1e-10))
        psnr = 10 * np.log10(255 ** 2 / mse) if mse != 0 else 0
        md = np.max(np.abs(original_arr - processed_arr))

        print(f"Mean Square Error (MSE): {mse}")
        print(f"Peak Mean Square Error (PMSE): {pmse}")
        print(f"Signal to Noise Ratio (SNR): {snr} dB")
        print(f"Peak Signal to Noise Ratio (PSNR): {psnr} dB")
        print(f"Maximum Difference (MD): {md}")

    except Exception as e:
        print(f"Error occurred while calculating similarity measures: {e}")


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

elif args.command == 'alpha_trimmed':
    if args.input and args.output and args.alpha is not None:
        apply_alpha_trimmed(args.input, args.output, args.alpha)
    else:
        print("Please provide input and output image files, and an alpha value")
        
elif args.command == "evaluate":
    calculate_similarity_measures(args.input, args.output)

elif args.command == 'help':
    parser.print_help()
else:
    print("Invalid command. Use --help for available commands.")


