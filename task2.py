import argparse
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description="Image processing tool for spatial domain operations")
parser.add_argument('--command', type=str, help="Command to run (e.g., histogram, hhyper, image_characteristics, linear_filter, non_linear_filter)")
parser.add_argument('--input', type=str, help="Input BMP image file")
parser.add_argument('--output', type=str, help="Output BMP image file")
parser.add_argument('--channel', type=int, choices=[0, 1, 2], help="Color channel to process (0: Red, 1: Green, 2: Blue)")
parser.add_argument('--direction', type=str, choices=['N', 'NE', 'E', 'SE'], help="Direction for linear filters")
args = parser.parse_args()


def read_image(input_file):
    try:
        return np.array(Image.open(input_file))
    except Exception as e:
        print(f"Error reading image: {e}")
        return None


def save_image(output_file, image_array):
    try:
        Image.fromarray(image_array.astype(np.uint8)).save(output_file)
        print(f"Image saved as {output_file}")
    except Exception as e:
        print(f"Error saving image: {e}")


# Histogram calculation (manual implementation)
def calculate_histogram(image, channel):
    try:
        histogram = [0] * 256
        channel_data = image[:, :, channel]
        for value in channel_data.flatten():
            histogram[value] += 1
        return histogram
    except Exception as e:
        print(f"Error calculating histogram: {e}")
        return None


def save_histogram(histogram, output_file):
    try:
        max_height = max(histogram)
        width = 256
        height = 200
        bar_width = 1
        scale = height / max_height

        histogram_image = np.full((height, width), 255, dtype=np.uint8)
        for x, freq in enumerate(histogram):
            bar_height = int(freq * scale)
            histogram_image[height - bar_height:height, x * bar_width: (x + 1) * bar_width] = 0

        save_image(output_file, histogram_image)
    except Exception as e:
        print(f"Error saving histogram: {e}")


# Hyperbolic histogram modification
def hyperbolic_modification(image, channel):
    try:
        channel_data = image[:, :, channel]
        modified_data = np.where(channel_data < 255, (channel_data ** 2) / (255 - channel_data + 1), 255)
        image[:, :, channel] = np.clip(modified_data, 0, 255).astype(np.uint8)
        return image
    except Exception as e:
        print(f"Error applying hyperbolic modification: {e}")
        return image


# Image characteristics calculation
def calculate_characteristics(image, channel):
    try:
        channel_data = image[:, :, channel].astype(np.float64)
        mean = np.mean(channel_data)
        variance = np.var(channel_data)
        std_dev = np.sqrt(variance)
        coeff_variation = std_dev / mean if mean != 0 else 0
        asymmetry = np.mean((channel_data - mean) ** 3) / (std_dev ** 3) if std_dev != 0 else 0
        flattening = np.mean((channel_data - mean) ** 4) / (std_dev ** 4) if std_dev != 0 else 0
        normalized = channel_data / 255
        entropy = -np.sum(normalized * np.log2(normalized + 1e-10))
        return {
            "mean": mean,
            "variance": variance,
            "std_dev": std_dev,
            "coeff_variation": coeff_variation,
            "asymmetry": asymmetry,
            "flattening": flattening,
            "entropy": entropy,
        }
    except Exception as e:
        print(f"Error calculating characteristics: {e}")
        return {}


# Linear filtering (manual convolution implementation)
def linear_filter(image, direction):
    try:
        kernel_map = {
            'N': np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]),
            'NE': np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]),
            'E': np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]]),
            'SE': np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
        }
        kernel = kernel_map[direction]
        filtered = np.zeros_like(image, dtype=np.float32)
        for channel in range(image.shape[2]):
            filtered[:, :, channel] = manual_convolution(image[:, :, channel], kernel)
        return np.clip(filtered, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f"Error applying linear filter: {e}")
        return image


def manual_convolution(channel_data, kernel):
    kernel_height, kernel_width = kernel.shape
    padded = np.pad(channel_data, pad_width=((1, 1), (1, 1)), mode='wrap')
    output = np.zeros_like(channel_data, dtype=np.float32)
    for i in range(channel_data.shape[0]):
        for j in range(channel_data.shape[1]):
            region = padded[i:i + kernel_height, j:j + kernel_width]
            output[i, j] = np.sum(region * kernel)
    return output


# Non-linear filtering (Kirsch operator)
def kirsch_operator(image):
    try:
        kernels = [
            np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),  # East
            np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),  # North-East
            np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),  # North
            np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])   # North-West
        ]
        filtered = np.zeros_like(image, dtype=np.float32)
        for channel in range(image.shape[2]):
            max_response = np.zeros_like(image[:, :, channel], dtype=np.float32)
            for kernel in kernels:
                response = manual_convolution(image[:, :, channel], kernel)
                max_response = np.maximum(max_response, response)
            filtered[:, :, channel] = max_response
        return np.clip(filtered, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f"Error applying Kirsch operator: {e}")
        return image


# Command handling
if args.command == 'histogram':
    if args.input and args.output and args.channel is not None:
        image = read_image(args.input)
        histogram = calculate_histogram(image, args.channel)
        save_histogram(histogram, args.output)
elif args.command == 'hhyper':
    if args.input and args.output and args.channel is not None:
        image = read_image(args.input)
        modified_image = hyperbolic_modification(image, args.channel)
        save_image(args.output, modified_image)
elif args.command == 'image_characteristics':
    if args.input and args.channel is not None:
        image = read_image(args.input)
        characteristics = calculate_characteristics(image, args.channel)
        for key, value in characteristics.items():
            print(f"{key}: {value}")
elif args.command == 'linear_filter':
    if args.input and args.output and args.direction:
        image = read_image(args.input)
        filtered_image = linear_filter(image, args.direction)
        save_image(args.output, filtered_image)
elif args.command == 'non_linear_filter':
    if args.input and args.output:
        image = read_image(args.input)
        filtered_image = kirsch_operator(image)
        save_image(args.output, filtered_image)
else:
    print("Invalid command or missing parameters. Use --help for usage information.")
