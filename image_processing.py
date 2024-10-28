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
parser.add_argument('--alpha', type=int, help="Alpha parameter for Alpha-trimmed mean filter (must be even)")
parser.add_argument('--q', type=float, help="Q parameter for Contraharmonic mean filter")

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

def contraharmonic_mean_filter(image, window_size=3, Q=1.5):
    padding = window_size // 2
    if image.ndim == 2:  # Grayscale image
        padded_image = np.pad(image, padding, mode='constant', constant_values=0)
        output_image = np.zeros_like(image, dtype=float)

        for x in range(padding, padded_image.shape[0] - padding):
            for y in range(padding, padded_image.shape[1] - padding):
                window = padded_image[x - padding:x + padding + 1, y - padding:y + padding + 1]
                
                # Only consider non-zero values in the window to avoid divide-by-zero issues
                non_zero_window = window[window != 0]
                
                if non_zero_window.size > 0:  # Check if there are any non-zero values
                    numerator = np.sum(non_zero_window ** (Q + 1))
                    denominator = np.sum(non_zero_window ** Q)
                    output_image[x - padding, y - padding] = numerator / denominator if denominator != 0 else 0
                else:
                    output_image[x - padding, y - padding] = 0  # If all values are zero, set output to zero

    else:  # Color image
        output_image = np.zeros_like(image, dtype=float)
        for c in range(image.shape[2]):  # Loop over color channels
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
                        output_image[x - padding, y - padding, c] = 0  # If all values are zero, set output to zero
                    
    return np.clip(output_image, 0, 255)

# Function to apply Alpha-trimmed mean filter and save result
def apply_alpha_trimmed(input_file, output_file, alpha):
    im = Image.open(input_file)
    arr = np.array(im)  # Convert to grayscale
    filtered_image = alpha_trimmed_mean_filter(arr, alpha=alpha)
    new_im = Image.fromarray(filtered_image.astype(np.uint8))
    new_im.save(output_file)
    print(f"Alpha-trimmed mean filtered image saved as {output_file}")

# Function to apply Contraharmonic Mean Filter and save result
def apply_contraharmonic(input_file, output_file, Q):
    im = Image.open(input_file)
    arr = np.array(im)
    filtered_image = contraharmonic_mean_filter(arr, Q=Q)
    new_im = Image.fromarray(filtered_image.astype(np.uint8))
    new_im.save(output_file)
    print(f"Contraharmonic mean filtered image saved as {output_file}")


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

elif args.command == 'alpha':
    if args.input and args.output and args.alpha is not None:
        apply_alpha_trimmed(args.input, args.output, args.alpha)
    else:
        print("Please provide input and output image files, and an alpha value")

elif args.command == 'cmean':
    if args.input and args.output and args.q is not None:
        apply_contraharmonic(args.input, args.output, args.q)
    else:
        print("Please provide input and output image files, and a Q value for the Contraharmonic filter")
        

elif args.command == 'help':
    parser.print_help()
else:
    print("Invalid command. Use --help for available commands.")


