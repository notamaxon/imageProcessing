import argparse
import time
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import math

parser = argparse.ArgumentParser(description="Image processing tool for spatial domain operations")
parser.add_argument('--command', type=str, help="Command to run (e.g., histogram, hhyper, image_characteristics, linear_filter, non_linear_filter)")
parser.add_argument('--input', type=str, help="Input BMP image file")
parser.add_argument('--output', type=str, help="Output BMP image file")
parser.add_argument('--channel', type=int, choices=[0, 1, 2], help="Color channel to process (0: Red, 1: Green, 2: Blue)")
parser.add_argument("--gmin", type=float, default=0, help="Minimum brightness in output image")
parser.add_argument("--gmax", type=float, default=255, help="Maximum brightness in output image")
parser.add_argument('--direction', type=str, choices=['N', 'NE', 'E', 'SE'], help="Direction for linear filters")
parser.add_argument('--cmean', action='store_true', help="Calculate mean")
parser.add_argument('--cvariance', action='store_true', help="Calculate variance")
parser.add_argument('--cstdev', action='store_true', help="Calculate standard deviation")
parser.add_argument('--cvarcoi', action='store_true', help="Calculate variation coefficient I")
parser.add_argument('--casyco', action='store_true', help="Calculate asymmetry coefficient")
parser.add_argument('--cflat', action='store_true', help="Calculate flattening coefficient")
parser.add_argument('--cvarcoii', action='store_true', help="Calculate variation coefficient II")
parser.add_argument('--centropy', action='store_true', help="Calculate information source entropy")
args = parser.parse_args()


def read_image(input_file):
    try:
        return np.array(Image.open(input_file))
    except Exception as e:
        print(f"Error reading image: {e}")
        return None


def save_image(output_file, image_array):
    try:
        image_array = np.clip(image_array, 0, 255)
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
    channel_data = image[:, :, channel]
    
    histogram = calculate_histogram(image, channel)
    N = channel_data.size  

    cumulative_histogram = np.zeros(256, dtype=np.float64)
    cumulative_histogram[0] = histogram[0]
    for i in range(1, 256):
        cumulative_histogram[i] = cumulative_histogram[i-1] + histogram[i]

    g_min = max(np.min(channel_data), 1)  
    g_max = max(np.max(channel_data), g_min + 1)  
    
    output_image = np.zeros_like(channel_data, dtype=np.float32)
    
    for f in range(256):
        
        cum_prob = cumulative_histogram[f] / N
        
        if g_min > 0 and g_max > g_min:
            try:
                g_f = g_min * np.exp(np.log(g_max / g_min) * cum_prob)
                
                output_image[channel_data == f] = np.clip(g_f, 0, 255)
            except Exception as e:
                print(f"Error processing intensity {f}: {e}")
                output_image[channel_data == f] = channel_data[channel_data == f]
    
    output_image = output_image.astype(np.uint8)
    
    modified_image = image.copy()
    modified_image[:, :, channel] = output_image
    
    return modified_image



def calculate_characteristics(image, channel):
    try:
        # Extract channel data
        channel_data = image[:, :, channel]
        
        # Calculate histogram manually
        histogram = [0] * 256
        for row in channel_data:
            for pixel in row:
                histogram[pixel] += 1
        
        # Total number of pixels
        N = channel_data.size
        
        # (C1) Mean calculation
        mean = 0
        for m in range(256):
            mean += m * histogram[m]
        mean /= N
        
        # (C2) Variance calculation
        variance = 0
        for m in range(256):
            variance += ((m - mean) ** 2) * histogram[m]
        variance /= N
        
        # Standard deviation
        std_dev = variance ** 0.5
        
        # (C2) Variation Coefficient I
        var_coeff_i = std_dev / mean if mean != 0 else 0
        
        # (C3) Asymmetry coefficient
        asymmetry_coeff = 0
        for m in range(256):
            asymmetry_coeff += ((m - mean) ** 3) * histogram[m]
        asymmetry_coeff /= (N * (std_dev ** 3))
        
        # (C4) Flattening coefficient
        flattening_coeff = 0
        for m in range(256):
            flattening_coeff += ((m - mean) ** 4) * histogram[m]
        flattening_coeff /= (N * (std_dev ** 4))
        flattening_coeff -= 3
        
        # (C5) Variation Coefficient II
        var_coeff_ii = 0
        for m in range(256):
            var_coeff_ii += (histogram[m] / N) ** 2
        
        # (C6) Information Source Entropy
        entropy = 0
        for m in range(256):
            # Avoid log(0) by adding a small epsilon
            if histogram[m] > 0:
                prob = histogram[m] / N
                entropy -= prob * math.log2(prob)
        
        # Return dictionary of characteristics
        return {
            "mean": mean,
            "variance": variance,
            "std_dev": std_dev,
            "var_coeff_i": var_coeff_i,
            "asymmetry_coeff": asymmetry_coeff,
            "flattening_coeff": flattening_coeff,
            "var_coeff_ii": var_coeff_ii,
            "entropy": entropy
        }
    
    except Exception as e:
        print(f"Error calculating characteristics: {e}")
        return None


# Linear filtering (manual convolution implementation)
def manual_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Universal convolution implementation for any mask.
    """
    kernel_height, kernel_width = kernel.shape
    pad_h, pad_w = kernel_height // 2, kernel_width // 2
    padded = np.pad(image, pad_width=((pad_h, pad_h), (pad_w, pad_w)), mode='edge')  # Edge padding
    output = np.zeros_like(image, dtype=np.float32)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract the region of interest
            region = padded[i:i + kernel_height, j:j + kernel_width]
            # Perform element-wise multiplication and summation
            output[i, j] = np.sum(region * kernel)
    
    return output

def optimized_edge_sharpening(image: np.ndarray) -> np.ndarray:
    """
    Optimized convolution for a specific edge-sharpening mask:
    h = [[1, -2, 1], [-2, 5, -2], [1, -2, 1]]
    """
    kernel = np.array([[1, -2, 1],
                       [-2, 5, -2],
                       [1, -2, 1]], dtype=np.float32)
    
    kernel_height, kernel_width = kernel.shape
    pad_h, pad_w = kernel_height // 2, kernel_width // 2
    padded = np.pad(image, pad_width=((pad_h, pad_h), (pad_w, pad_w)), mode='edge')  # Edge padding
    output = np.zeros_like(image, dtype=np.float32)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Manually compute the weighted sum for this specific mask
            center = padded[i + 1, j + 1] * 5
            top = padded[i, j + 1] * -2
            bottom = padded[i + 2, j + 1] * -2
            left = padded[i + 1, j] * -2
            right = padded[i + 1, j + 2] * -2
            top_left = padded[i, j]
            top_right = padded[i, j + 2]
            bottom_left = padded[i + 2, j]
            bottom_right = padded[i + 2, j + 2]
            
            output[i, j] = (center + top + bottom + left + right +
                            top_left + top_right + bottom_left + bottom_right)
    
    return output



def apply_manual_convolution(input_file, output_file, kernel):
    im = Image.open(input_file)
    arr = np.array(im)
    if len(arr.shape) == 3:  # Process color images
        filtered_image = np.zeros_like(arr, dtype=np.float32)
        for channel in range(arr.shape[2]):
            filtered_image[:, :, channel] = manual_convolution(arr[:, :, channel], kernel)
    else:  # Process grayscale images
        filtered_image = manual_convolution(arr, kernel)
    save_image(output_file, filtered_image)

def apply_optimized_edge_sharpening(input_file, output_file):
    im = Image.open(input_file)
    arr = np.array(im)
    if len(arr.shape) == 3:  # Process color images
        filtered_image = np.zeros_like(arr, dtype=np.float32)
        for channel in range(arr.shape[2]):
            filtered_image[:, :, channel] = optimized_edge_sharpening(arr[:, :, channel])
    else:  # Process grayscale images
        filtered_image = optimized_edge_sharpening(arr)
    save_image(output_file, filtered_image)


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
        
        if characteristics:
            if args.cmean:
                print(f"Mean: {characteristics['mean']}")
            if args.cvariance:
                print(f"Variance: {characteristics['variance']}")
            if args.cstdev:
                print(f"Standard Deviation: {characteristics['std_dev']}")
            if args.cvarcoi:
                print(f"Variation Coefficient I: {characteristics['var_coeff_i']}")
            if args.casyco:
                print(f"Asymmetry Coefficient: {characteristics['asymmetry_coeff']}")
            if args.cflat:
                print(f"Flattening Coefficient: {characteristics['flattening_coeff']}")
            if args.cvarcoii:
                print(f"Variation Coefficient II: {characteristics['var_coeff_ii']}")
            if args.centropy:
                print(f"Information Source Entropy: {characteristics['entropy']}")
                
elif args.command == 'manual_filter':
    if args.input and args.output:
        # Example kernel for edge sharpening
        universal_kernel = np.array([[0, -1, 0],
                                     [-1, 5, -1],
                                     [0, -1, 0]], dtype=np.float32)
        apply_manual_convolution(args.input, args.output, universal_kernel)
elif args.command == 'optimized_filter':
    if args.input and args.output:
        apply_optimized_edge_sharpening(args.input, args.output)
elif args.command == 'non_linear_filter':
    if args.input and args.output:
        image = read_image(args.input)
        filtered_image = kirsch_operator(image)
        save_image(args.output, filtered_image)
else:
    print("Invalid command or missing parameters. Use --help for usage information.")
