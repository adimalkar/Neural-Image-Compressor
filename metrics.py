import numpy as np
import math

def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between the original and compressed images.
    PSNR is a standard metric used to measure the quality of reconstruction of lossy compression codecs.
    Higher PSNR generally indicates that the reconstruction is of higher quality.
    
    Args:
        original (np.ndarray): The original uncompressed image array (RGB or Grayscale)
        compressed (np.ndarray): The reconstructed compressed image array
        
    Returns:
        float: PSNR value in decibels (dB), or float('inf') if images are identical
    """
    # Ensure images have same shape
    if original.shape != compressed.shape:
        raise ValueError("Original and compressed images must have the same dimensions.")
        
    # Convert arrays to float64 to prevent overflow when calculating square difference
    orig_float = original.astype(np.float64)
    comp_float = compressed.astype(np.float64)
    
    # Calculate Mean Squared Error (MSE)
    mse = np.mean((orig_float - comp_float) ** 2)
    
    # If MSE is zero, it means no noise is present in the signal
    if mse == 0:
        return float('inf')
        
    # Assume 8-bit image where maximum pixel value is 255
    max_pixel = 255.0
    
    # Calculate PSNR
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    
    return round(psnr, 4)

def calculate_compression_ratio(original_size_bytes: int, compressed_size_bytes: int) -> float:
    """
    Calculate the compression ratio achieved by the neural compressor.
    
    Args:
        original_size_bytes (int): Size of the original image
        compressed_size_bytes (int): Size of the compressed bitstream
        
    Returns:
        float: The compression ratio (e.g., 5.0 means the compressed file is 5x smaller)
    """
    if compressed_size_bytes == 0:
        raise ValueError("Compressed size cannot be zero.")
        
    return round(original_size_bytes / compressed_size_bytes, 2)


def calculate_bpp(compressed_size_bytes: int, width: int, height: int) -> float:
    """
    Calculate Bits Per Pixel (BPP), standard efficiency metric in image compression.
    
    Args:
        compressed_size_bytes (int): Size of the compressed bitstream in bytes
        width (int): Image width in pixels
        height (int): Image height in pixels
        
    Returns:
        float: BPP value representing bits consumed per pixel
    """
    num_pixels = width * height
    if num_pixels <= 0:
        raise ValueError("Image dimensions must be positive integers.")
        
    bpp = (compressed_size_bytes * 8.0) / num_pixels
    return round(bpp, 4)


def calculate_ssim(
    original: np.ndarray, 
    compressed: np.ndarray, 
    k1: float = 0.01, 
    k2: float = 0.03, 
    max_val: float = 255.0
) -> float:
    """
    Calculate Mean Structural Similarity Index Measure (SSIM) between two images.
    Measures perceived image degradation by evaluating luminance, contrast, and structure.
    
    Args:
        original (np.ndarray): Original image array
        compressed (np.ndarray): Compressed/reconstructed image array
        k1 (float): Small constant to stabilize division near zero luminance
        k2 (float): Small constant to stabilize division near zero contrast
        max_val (float): Dynamic range of pixel values (typically 255 for 8-bit)
        
    Returns:
        float: SSIM index ranging from -1 to 1 (1 indicates perfect structural match)
    """
    if original.shape != compressed.shape:
        raise ValueError("Input images must have the same shape for SSIM calculation.")
        
    orig = original.astype(np.float64)
    comp = compressed.astype(np.float64)
    
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    
    mu_x = np.mean(orig)
    mu_y = np.mean(comp)
    
    sigma_x_sq = np.var(orig)
    sigma_y_sq = np.var(comp)
    sigma_xy = np.cov(orig.flatten(), comp.flatten())[0, 1]
    
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x_sq + sigma_y_sq + c2)
    
    ssim = numerator / denominator
    return round(float(ssim), 4)


def calculate_mae(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE / L1 loss) between original and reconstructed images.
    Preserves sharper edges and texture detail compared to L2/MSE in perceptual neural codecs.
    
    Args:
        original (np.ndarray): Original image array
        compressed (np.ndarray): Reconstructed image array
        
    Returns:
        float: MAE value representing average absolute pixel deviation
    """
    if original.shape != compressed.shape:
        raise ValueError("Original and compressed images must have the same dimensions.")
        
    orig_float = original.astype(np.float64)
    comp_float = compressed.astype(np.float64)
    return round(float(np.mean(np.abs(orig_float - comp_float))), 4)


def calculate_rate_distortion(bpp: float, distortion_mse: float, lambda_param: float = 0.01) -> float:
    """
    Calculate Rate-Distortion cost: J = D + lambda * R.
    Standard optimization objective in learned neural image compression trade-offs.
    
    Args:
        bpp (float): Rate in bits per pixel
        distortion_mse (float): Distortion measured via Mean Squared Error (MSE)
        lambda_param (float): Lagrange multiplier controlling rate vs quality trade-off
        
    Returns:
        float: Combined Rate-Distortion objective value
    """
    if bpp < 0 or distortion_mse < 0:
        raise ValueError("Rate (bpp) and distortion (MSE) must be non-negative.")
        
    cost = distortion_mse + (lambda_param * bpp)
    return round(float(cost), 6)

