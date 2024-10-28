import argparse
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description="Image reading and writing tool using NumPy")
parser.add_argument('--command', type=str, help="Command to run (e.g., readwrite, help)")
parser.add_argument('--input', type=str, help="Input BMP image file")
parser.add_argument('--output', type=str, help="Output BMP image file")
parser.add_argument('--brightness', type=int, help="Brightness adjustment value (positive to increase, negative to decrease)")
parser.add_argument('--contrast', type=int, help="Contrast adjustment value (negative to decrease, positive to increase)")
parser.add_argument('--alpha', type=int, help="Alpha parameter for Alpha-trimmed mean filter (must be even)")

args = parser.parse_args()


def read_and_write_image(input_file, output_file, brightness_value):
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

elif args.command == 'alpha_trimmed':
    if args.input and args.output and args.alpha is not None:
        apply_alpha_trimmed(args.input, args.output, args.alpha)
    else:
        print("Please provide input and output image files, and an alpha value")
        
elif args.command == 'help':
    parser.print_help()
else:
    print("Invalid command. Use --help for available commands.")


