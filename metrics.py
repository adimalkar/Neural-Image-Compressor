import numpy as np
import math
from typing import Dict, Any, List, Tuple

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


def calculate_bd_rate(
    rate1: List[float], psnr1: List[float],
    rate2: List[float], psnr2: List[float]
) -> float:
    """
    Calculate Bjontegaard Delta Rate (BD-Rate / BD-BR).
    Standard ITU-T metric measuring the average percentage bitrate difference between
    two Rate-Distortion curves for equivalent PSNR objective reconstruction quality.
    A negative BD-rate indicates that Curve 2 is more efficient (uses less bitrate) than Curve 1.
    """
    if len(rate1) != len(psnr1) or len(rate2) != len(psnr2):
        raise ValueError("Rate and PSNR arrays must have identical lengths.")
    if len(rate1) < 4 or len(rate2) < 4:
        raise ValueError("At least 4 operational RD points are required for Bjontegaard curve fitting.")

    # Convert rates to natural log space
    lrate1 = np.log(np.array(rate1, dtype=np.float64))
    lrate2 = np.log(np.array(rate2, dtype=np.float64))
    d1 = np.array(psnr1, dtype=np.float64)
    d2 = np.array(psnr2, dtype=np.float64)

    # Fit 3rd-order polynomials: log_rate = f(psnr)
    p1 = np.polyfit(d1, lrate1, 3)
    p2 = np.polyfit(d2, lrate2, 3)

    # Integration interval: overlap of PSNR ranges
    min_int = max(np.min(d1), np.min(d2))
    max_int = min(np.max(d1), np.max(d2))

    if min_int >= max_int:
        raise ValueError("No overlapping PSNR range between the two Rate-Distortion curves.")

    # Indefinite integrals of the polynomials
    p_int1 = np.polyint(p1)
    p_int2 = np.polyint(p2)

    # Definite integral over the shared range
    int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
    int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)

    # Average difference in log-rate
    avg_diff = (int2 - int1) / (max_int - min_int)

    # Convert difference to percentage BD-Rate savings
    bd_rate_pct = (math.exp(avg_diff) - 1.0) * 100.0
    return round(float(bd_rate_pct), 2)


def evaluate_compression_suite(
    original: np.ndarray,
    compressed: np.ndarray,
    compressed_bytes: int,
    lambda_param: float = 0.01
) -> Dict[str, Any]:
    """
    Computes a full battery of neural image compression metrics for an image pair.
    """
    if original.shape != compressed.shape:
        raise ValueError("Original and compressed images must have identical shapes.")

    h, w = original.shape[:2]
    orig_bytes = original.nbytes

    psnr_val = calculate_psnr(original, compressed)
    ssim_val = calculate_ssim(original, compressed)
    mae_val = calculate_mae(original, compressed)
    bpp_val = calculate_bpp(compressed_bytes, w, h)
    ratio_val = calculate_compression_ratio(orig_bytes, compressed_bytes)

    orig_float = original.astype(np.float64)
    comp_float = compressed.astype(np.float64)
    mse_val = float(np.mean((orig_float - comp_float) ** 2))
    rd_cost = calculate_rate_distortion(bpp_val, mse_val, lambda_param=lambda_param)

    # Quality grading heuristic
    if psnr_val >= 38.0 and ssim_val >= 0.95:
        grade = "EXCELLENT"
    elif psnr_val >= 32.0 and ssim_val >= 0.88:
        grade = "GOOD"
    elif psnr_val >= 26.0:
        grade = "FAIR"
    else:
        grade = "POOR"

    return {
        "psnr_db": psnr_val,
        "ssim": ssim_val,
        "mae": mae_val,
        "mse": round(mse_val, 4),
        "bpp": bpp_val,
        "compression_ratio": ratio_val,
        "rd_cost": rd_cost,
        "perceptual_grade": grade,
        "resolution": f"{w}x{h}"
    }

