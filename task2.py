import argparse
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

parser = argparse.ArgumentParser(description="Image processing tool for spatial domain operations")
parser.add_argument('--command', type=str, help="Command to run (e.g., histogram, hhyper, image_characteristics, linear_filter, non_linear_filter)")
parser.add_argument('--input', type=str, help="Input BMP image file")
parser.add_argument('--output', type=str, help="Output BMP image file")
parser.add_argument('--channel', type=int, choices=[0, 1, 2], help="Color channel to process (0: Red, 1: Green, 2: Blue)")
parser.add_argument("--gmin", type=float, default=0, help="Minimum brightness in output image")
parser.add_argument("--gmax", type=float, default=255, help="Maximum brightness in output image")
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


def save_histogram(histogram, output_file, channel):
    try:
        # Histogram visualization parameters
        width = 800
        height = 400
        margin = 60 
        
        histogram_image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(histogram_image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 12)
            title_font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
        
        max_frequency = max(histogram)
        
        if channel == 0:  
            color = (255, 0, 0)
            channel_name = "Red"
        elif channel == 1:  
            color = (0, 255, 0)
            channel_name = "Green"
        elif channel == 2:  
            color = (0, 0, 255)
            channel_name = "Blue"
        else:
            color = (0, 0, 0)
            channel_name = "Unknown"
        
        # X-axis
        draw.line([(margin, height - margin), (width - margin, height - margin)], fill='black', width=2)
        # Y-axis
        draw.line([(margin, height - margin), (margin, margin)], fill='black', width=2)
        
        plot_width = width - 2 * margin
        plot_height = height - 2 * margin
        bar_width = plot_width // 256
        
        for x, freq in enumerate(histogram):
            scaled_height = int((freq / max_frequency) * plot_height)
            
            x_start = margin + x * bar_width
            y_start = height - margin - scaled_height
            
            draw.rectangle([x_start, y_start, x_start + bar_width, height - margin], 
                           fill=color, outline=color)
        
        x_axis_labels = [0, 51, 102, 153, 204, 255]
        plot_width = width - 2 * margin
        for label in x_axis_labels:

            x_position = margin + (label / 255) * plot_width
            draw.text((x_position - 10, height - margin + 10), str(label), fill='black', font=font)
        
        y_axis_steps = 5
        for i in range(y_axis_steps + 1):
            y_position = height - margin - i * (plot_height // y_axis_steps)
            label = f"{int(max_frequency * i / y_axis_steps):,}"
            draw.text((10, y_position - 5), label, fill='black', font=font)
        
        draw.text((width // 2 - 50, height - 20), "Pixel Intensity", fill='black', font=font)
        draw.text((20, height // 2), "Frequency", fill='black', font=font, anchor='ms')
        
        draw.text((width // 2, 30), f"{channel_name} Channel Histogram", 
                  fill='black', font=title_font, anchor='mt')
        
        histogram_image.save(output_file)
        print(f"Histogram saved as {output_file}")
        
    except Exception as e:
        print(f"Error saving histogram: {e}")



# Hyperbolic histogram modification
def hyperbolic_modification(image, channel):
    # Extract the specific channel
    channel_data = image[:, :, channel]
    
    # Calculate histogram
    histogram = calculate_histogram(image, channel)
    N = channel_data.size  # Total number of pixels

    # Calculate cumulative histogram
    cumulative_histogram = np.cumsum(histogram)
    
    # Automatically determine g_min and g_max
    # g_min will be the minimum pixel value in the image
    # g_max will be the maximum pixel value in the image
    g_min = np.min(channel_data)
    g_max = np.max(channel_data)

    # Create output image
    output_image = np.zeros_like(channel_data, dtype=np.float32)
    
    # Apply hyperbolic modification using the provided formula
    for f in range(256):
        # Calculate cumulative probability
        cum_prob = cumulative_histogram[f] / N
        
        # Apply hyperbolic transformation
        g_f = g_min * ((g_max / g_min) ** cum_prob)
        
        # Map pixels with this original value
        output_image[channel_data == f] = g_f

    # Normalize and clip the output image
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    
    # Reconstruct the full image with the modified channel
    modified_image = image.copy()
    modified_image[:, :, channel] = output_image
    
    return modified_image



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
        save_histogram(histogram, args.output, args.channel)
elif args.command == 'hhyper':
    if args.input and args.output and args.channel is not None:
        image = read_image(args.input)
        # Ensure gmin and gmax are passed to the function
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
