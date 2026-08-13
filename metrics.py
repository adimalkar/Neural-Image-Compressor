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
