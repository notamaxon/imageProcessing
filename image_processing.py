import argparse
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description="Image reading and writing tool using NumPy")
parser.add_argument('--command', type=str, help="Command to run (e.g., readwrite, help)")
parser.add_argument('--input', type=str, help="Input BMP image file")
parser.add_argument('--output', type=str, help="Output BMP image file")
parser.add_argument('--brightness', type=int, help="Brightness adjustment value (positive to increase, negative to decrease)")
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
elif args.command == 'help':
    parser.print_help()
else:
    print("Invalid command. Use --help for available commands.")


