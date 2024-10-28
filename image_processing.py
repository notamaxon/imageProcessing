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


elif args.command == 'help':
    parser.print_help()
else:
    print("Invalid command. Use --help for available commands.")


