"""
Image quality analysis utilities for deepfake dataset.

This module provides functions for analyzing image quality metrics
when actual image data is available.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# OpenCV is optional for image quality analysis
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def calculate_laplacian_variance(image_path: str) -> float:
    """Calculate Laplacian variance as a blur metric.
    
    Higher values indicate sharper images, lower values indicate blurrier images.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Laplacian variance score (float)
    """
    if not CV2_AVAILABLE:
        # Fallback: use PIL-based edge detection approximation
        try:
            with Image.open(image_path) as img:
                img_gray = img.convert('L')
                # Simple gradient approximation
                pixels = np.array(img_gray)
                gradient_x = np.abs(np.diff(pixels, axis=1))
                gradient_y = np.abs(np.diff(pixels, axis=0))
                edge_strength = np.mean(gradient_x) + np.mean(gradient_y)
                return float(edge_strength)
        except Exception:
            return 0.0
    
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return cv2.Laplacian(img, cv2.CV_64F).var()
    except Exception:
        return 0.0


def calculate_brightness_contrast(image_path: str) -> Tuple[float, float]:
    """Calculate brightness and contrast metrics.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (brightness, contrast) values
    """
    try:
        if CV2_AVAILABLE:
            img = cv2.imread(str(image_path))
            if img is None:
                return 0.0, 0.0
            # Convert to grayscale for brightness/contrast calculation
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            # Fallback to PIL
            with Image.open(image_path) as img:
                gray = img.convert('L')
                gray = np.array(gray)
        
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        return float(brightness), float(contrast)
    except Exception:
        return 0.0, 0.0


def get_image_resolution(image_path: str) -> Tuple[int, int]:
    """Get image resolution (width, height).
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (width, height)
    """
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return (0, 0)


def calculate_aspect_ratio(width: int, height: int) -> float:
    """Calculate aspect ratio from width and height.
    
    Args:
        width: Image width
        height: Image height
        
    Returns:
        Aspect ratio (width/height)
    """
    if height == 0:
        return 0.0
    return width / height


def get_file_size(image_path: str) -> int:
    """Get file size in bytes.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        File size in bytes
    """
    try:
        return Path(image_path).stat().st_size
    except Exception:
        return 0


def analyze_image_quality(image_path: str) -> Dict[str, float]:
    """Perform comprehensive image quality analysis.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary containing quality metrics
    """
    quality_metrics = {
        'laplacian_variance': 0.0,
        'brightness': 0.0,
        'contrast': 0.0,
        'width': 0,
        'height': 0,
        'aspect_ratio': 0.0,
        'file_size': 0,
        'is_valid': False
    }
    
    try:
        # Basic validity check
        with Image.open(image_path) as img:
            img.verify()
        
        # Calculate metrics
        quality_metrics['laplacian_variance'] = calculate_laplacian_variance(image_path)
        quality_metrics['brightness'], quality_metrics['contrast'] = calculate_brightness_contrast(image_path)
        quality_metrics['width'], quality_metrics['height'] = get_image_resolution(image_path)
        quality_metrics['aspect_ratio'] = calculate_aspect_ratio(quality_metrics['width'], quality_metrics['height'])
        quality_metrics['file_size'] = get_file_size(image_path)
        quality_metrics['is_valid'] = True
        
    except Exception as e:
        # Image is invalid or corrupted
        quality_metrics['error'] = str(e)
    
    return quality_metrics


def classify_image_quality(metrics: Dict[str, float], 
                          blur_threshold: float = 100.0,
                          min_resolution: Tuple[int, int] = (256, 256)) -> str:
    """Classify image quality based on metrics.
    
    Args:
        metrics: Dictionary of quality metrics
        blur_threshold: Laplacian variance threshold for blur classification
        min_resolution: Minimum acceptable resolution (width, height)
        
    Returns:
        Quality classification: 'HIGH', 'MEDIUM', or 'LOW'
    """
    if not metrics.get('is_valid', False):
        return 'INVALID'
    
    quality_score = 0
    
    # Blur assessment
    if metrics['laplacian_variance'] > blur_threshold:
        quality_score += 1
    elif metrics['laplacian_variance'] > blur_threshold / 2:
        quality_score += 0.5
    
    # Resolution assessment
    if metrics['width'] >= min_resolution[0] and metrics['height'] >= min_resolution[1]:
        quality_score += 1
    elif metrics['width'] >= min_resolution[0] / 2 and metrics['height'] >= min_resolution[1] / 2:
        quality_score += 0.5
    
    # Contrast assessment
    if metrics['contrast'] > 50:
        quality_score += 1
    elif metrics['contrast'] > 25:
        quality_score += 0.5
    
    # Final classification
    if quality_score >= 2.5:
        return 'HIGH'
    elif quality_score >= 1.5:
        return 'MEDIUM'
    else:
        return 'LOW'


def batch_analyze_image_quality(image_paths: List[str], 
                                max_samples: Optional[int] = None) -> List[Dict[str, float]]:
    """Analyze quality for multiple images.
    
    Args:
        image_paths: List of image file paths
        max_samples: Maximum number of samples to analyze (None for all)
        
    Returns:
        List of quality metric dictionaries
    """
    if max_samples:
        image_paths = image_paths[:max_samples]
    
    results = []
    for image_path in image_paths:
        metrics = analyze_image_quality(image_path)
        metrics['path'] = image_path
        results.append(metrics)
    
    return results


def summarize_quality_results(quality_results: List[Dict[str, float]]) -> Dict[str, any]:
    """Summarize quality analysis results.
    
    Args:
        quality_results: List of quality metric dictionaries
        
    Returns:
        Summary statistics
    """
    if not quality_results:
        return {}
    
    valid_results = [r for r in quality_results if r.get('is_valid', False)]
    invalid_count = len(quality_results) - len(valid_results)
    
    if not valid_results:
        return {
            'total_samples': len(quality_results),
            'valid_samples': 0,
            'invalid_samples': invalid_count,
            'invalid_percentage': (invalid_count / len(quality_results) * 100) if quality_results else 0.0
        }
    
    # Classify quality
    quality_classifications = [
        classify_image_quality(metrics) for metrics in valid_results
    ]
    
    quality_counts = {
        'HIGH': quality_classifications.count('HIGH'),
        'MEDIUM': quality_classifications.count('MEDIUM'),
        'LOW': quality_classifications.count('LOW'),
        'INVALID': invalid_count
    }
    
    # Calculate statistics for numeric metrics
    laplacian_values = [r['laplacian_variance'] for r in valid_results]
    brightness_values = [r['brightness'] for r in valid_results]
    contrast_values = [r['contrast'] for r in valid_results]
    
    summary = {
        'total_samples': len(quality_results),
        'valid_samples': len(valid_results),
        'invalid_samples': invalid_count,
        'invalid_percentage': invalid_count / len(quality_results) * 100,
        'quality_distribution': quality_counts,
        'laplacian_stats': {
            'mean': float(np.mean(laplacian_values)),
            'std': float(np.std(laplacian_values)),
            'min': float(np.min(laplacian_values)),
            'max': float(np.max(laplacian_values))
        },
        'brightness_stats': {
            'mean': float(np.mean(brightness_values)),
            'std': float(np.std(brightness_values)),
            'min': float(np.min(brightness_values)),
            'max': float(np.max(brightness_values))
        },
        'contrast_stats': {
            'mean': float(np.mean(contrast_values)),
            'std': float(np.std(contrast_values)),
            'min': float(np.min(contrast_values)),
            'max': float(np.max(contrast_values))
        },
        'cv2_available': CV2_AVAILABLE
    }
    
    return summary