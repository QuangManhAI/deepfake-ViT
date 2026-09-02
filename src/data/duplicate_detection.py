"""
Duplicate detection utilities for deepfake dataset.

This module provides functions for detecting exact and near-duplicate images
using various hashing and similarity methods.
"""

import hashlib
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict


def calculate_file_hash(image_path: str, hash_algorithm: str = 'md5') -> str:
    """Calculate file hash for exact duplicate detection.
    
    Args:
        image_path: Path to the image file
        hash_algorithm: Hash algorithm to use ('md5', 'sha1', 'sha256')
        
    Returns:
        Hexadecimal hash string
    """
    hash_func = hashlib.new(hash_algorithm)
    try:
        with open(image_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    except Exception:
        return ""


def calculate_perceptual_hash(image_path: str, hash_size: int = 8) -> str:
    """Calculate perceptual hash (pHash) for near-duplicate detection.
    
    Args:
        image_path: Path to the image file
        hash_size: Size of the hash (default 8x8)
        
    Returns:
        Hexadecimal hash string representing perceptual signature
    """
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale and resize
            img = img.convert('L').resize((hash_size, hash_size), Image.LANCZOS)
            
            # Convert to numpy array
            pixels = np.array(img)
            
            # Calculate DCT (simplified - using average hash for performance)
            # For production, consider using scipy.fftpack.dct for true pHash
            avg = pixels.mean()
            
            # Generate hash based on comparison to average
            hash_bits = (pixels > avg).flatten()
            hash_string = ''.join(['1' if bit else '0' for bit in hash_bits])
            
            # Convert to hexadecimal
            hash_hex = hex(int(hash_string, 2))[2:].zfill(len(hash_string) // 4)
            
            return hash_hex
    except Exception:
        return ""


def calculate_average_hash(image_path: str, hash_size: int = 8) -> str:
    """Calculate average hash (aHash) for near-duplicate detection.
    
    Args:
        image_path: Path to the image file
        hash_size: Size of the hash (default 8x8)
        
    Returns:
        Hexadecimal hash string
    """
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale and resize
            img = img.convert('L').resize((hash_size, hash_size), Image.LANCZOS)
            
            # Convert to numpy array
            pixels = np.array(img)
            
            # Calculate average
            avg = pixels.mean()
            
            # Generate hash
            hash_bits = (pixels > avg).flatten()
            hash_string = ''.join(['1' if bit else '0' for bit in hash_bits])
            
            # Convert to hexadecimal
            hash_hex = hex(int(hash_string, 2))[2:].zfill(len(hash_string) // 4)
            
            return hash_hex
    except Exception:
        return ""


def calculate_difference_hash(image_path: str, hash_size: int = 8) -> str:
    """Calculate difference hash (dHash) for near-duplicate detection.
    
    Args:
        image_path: Path to the image file
        hash_size: Size of the hash (default 8x8)
        
    Returns:
        Hexadecimal hash string
    """
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale and resize
            img = img.convert('L').resize((hash_size + 1, hash_size), Image.LANCZOS)
            
            # Convert to numpy array
            pixels = np.array(img)
            
            # Calculate differences between adjacent pixels
            diff = pixels[:, 1:] > pixels[:, :-1]
            
            # Generate hash
            hash_bits = diff.flatten()
            hash_string = ''.join(['1' if bit else '0' for bit in hash_bits])
            
            # Convert to hexadecimal
            hash_hex = hex(int(hash_string, 2))[2:].zfill(len(hash_string) // 4)
            
            return hash_hex
    except Exception:
        return ""


def hamming_distance(hash1: str, hash2: str) -> int:
    """Calculate Hamming distance between two hash strings.
    
    Args:
        hash1: First hash string
        hash2: Second hash string
        
    Returns:
        Number of differing bits
    """
    if len(hash1) != len(hash2):
        return len(hash1)  # Maximum possible distance
    
    # Convert to binary and count differences
    try:
        h1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
        h2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)
        return sum(c1 != c2 for c1, c2 in zip(h1, h2))
    except Exception:
        return len(hash1)


def find_exact_duplicates(image_paths: List[str]) -> Dict[str, List[str]]:
    """Find exact duplicates using file hash.
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        Dictionary mapping hash to list of file paths with that hash
    """
    hash_map = defaultdict(list)
    
    for image_path in image_paths:
        file_hash = calculate_file_hash(image_path)
        if file_hash:
            hash_map[file_hash].append(image_path)
    
    # Return only entries with duplicates
    duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
    return duplicates


def find_near_duplicates(image_paths: List[str], 
                        hash_function: str = 'average_hash',
                        threshold: int = 5) -> Dict[str, List[str]]:
    """Find near-duplicates using perceptual hashing.
    
    Args:
        image_paths: List of image file paths
        hash_function: Hash function to use ('average_hash', 'perceptual_hash', 'difference_hash')
        threshold: Maximum Hamming distance to consider as near-duplicate
        
    Returns:
        Dictionary mapping canonical image to list of near-duplicates
    """
    # Select hash function
    hash_func = {
        'average_hash': calculate_average_hash,
        'perceptual_hash': calculate_perceptual_hash,
        'difference_hash': calculate_difference_hash
    }.get(hash_function, calculate_average_hash)
    
    # Calculate hashes for all images
    hash_map = {}
    for image_path in image_paths:
        img_hash = hash_func(image_path)
        if img_hash:
            hash_map[image_path] = img_hash
    
    # Find near-duplicates by comparing hashes
    near_duplicates = defaultdict(list)
    processed = set()
    
    for img1, hash1 in hash_map.items():
        if img1 in processed:
            continue
            
        duplicates = [img1]
        processed.add(img1)
        
        for img2, hash2 in hash_map.items():
            if img2 in processed:
                continue
                
            distance = hamming_distance(hash1, hash2)
            if distance <= threshold:
                duplicates.append(img2)
                processed.add(img2)
        
        if len(duplicates) > 1:
            canonical = duplicates[0]  # Use first as canonical
            near_duplicates[canonical] = duplicates[1:]
    
    return dict(near_duplicates)


def analyze_duplicate_statistics(duplicates: Dict[str, List[str]]) -> Dict[str, any]:
    """Analyze statistics of duplicate detection results.
    
    Args:
        duplicates: Dictionary of duplicates (canonical -> duplicates list)
        
    Returns:
        Statistics about duplicates
    """
    if not duplicates:
        return {
            'total_duplicate_groups': 0,
            'total_duplicate_files': 0,
            'average_group_size': 0.0,
            'max_group_size': 0
        }
    
    group_sizes = [len(dup_list) + 1 for dup_list in duplicates.values()]  # +1 for canonical
    total_duplicate_files = sum(group_sizes)
    
    return {
        'total_duplicate_groups': len(duplicates),
        'total_duplicate_files': total_duplicate_files,
        'average_group_size': np.mean(group_sizes),
        'max_group_size': max(group_sizes),
        'group_size_distribution': group_sizes
    }


def detect_leakage_by_hash(train_paths: List[str], 
                          val_paths: List[str], 
                          test_paths: List[str],
                          hash_function: str = 'file_hash') -> Dict[str, any]:
    """Detect data leakage between splits using hash comparison.
    
    Args:
        train_paths: List of training image paths
        val_paths: List of validation image paths
        test_paths: List of test image paths
        hash_function: Hash function to use ('file_hash', 'average_hash', etc.)
        
    Returns:
        Leakage analysis results
    """
    # Select hash function
    if hash_function == 'file_hash':
        hash_func = calculate_file_hash
    else:
        hash_func = {
            'average_hash': calculate_average_hash,
            'perceptual_hash': calculate_perceptual_hash,
            'difference_hash': calculate_difference_hash
        }.get(hash_function, calculate_average_hash)
    
    # Calculate hashes for each split
    train_hashes = {}
    val_hashes = {}
    test_hashes = {}
    
    for path in train_paths:
        h = hash_func(path)
        if h:
            train_hashes[h] = path
    
    for path in val_paths:
        h = hash_func(path)
        if h:
            val_hashes[h] = path
    
    for path in test_paths:
        h = hash_func(path)
        if h:
            test_hashes[h] = path
    
    # Find leakage
    train_val_leakage = set(train_hashes.keys()) & set(val_hashes.keys())
    train_test_leakage = set(train_hashes.keys()) & set(test_hashes.keys())
    val_test_leakage = set(val_hashes.keys()) & set(test_hashes.keys())
    
    return {
        'train_val_leakage_count': len(train_val_leakage),
        'train_test_leakage_count': len(train_test_leakage),
        'val_test_leakage_count': len(val_test_leakage),
        'train_val_leakage_samples': [(train_hashes[h], val_hashes[h]) for h in train_val_leakage],
        'train_test_leakage_samples': [(train_hashes[h], test_hashes[h]) for h in train_test_leakage],
        'val_test_leakage_samples': [(val_hashes[h], test_hashes[h]) for h in val_test_leakage],
        'total_train_samples': len(train_hashes),
        'total_val_samples': len(val_hashes),
        'total_test_samples': len(test_hashes)
    }


def generate_duplicate_report(duplicates: Dict[str, List[str]], 
                           near_duplicates: Dict[str, List[str]]) -> str:
    """Generate a comprehensive duplicate detection report.
    
    Args:
        duplicates: Exact duplicates dictionary
        near_duplicates: Near-duplicates dictionary
        
    Returns:
        Formatted report string
    """
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("DUPLICATE DETECTION REPORT")
    report_lines.append("=" * 60)
    
    # Exact duplicates
    exact_stats = analyze_duplicate_statistics(duplicates)
    report_lines.append(f"\nEXACT DUPLICATES:")
    report_lines.append(f"  Duplicate groups: {exact_stats['total_duplicate_groups']}")
    report_lines.append(f"  Total duplicate files: {exact_stats['total_duplicate_files']}")
    report_lines.append(f"  Average group size: {exact_stats['average_group_size']:.2f}")
    report_lines.append(f"  Max group size: {exact_stats['max_group_size']}")
    
    # Near-duplicates
    near_stats = analyze_duplicate_statistics(near_duplicates)
    report_lines.append(f"\nNEAR-DUPLICATES:")
    report_lines.append(f"  Near-duplicate groups: {near_stats['total_duplicate_groups']}")
    report_lines.append(f"  Total near-duplicate files: {near_stats['total_duplicate_files']}")
    report_lines.append(f"  Average group size: {near_stats['average_group_size']:.2f}")
    report_lines.append(f"  Max group size: {near_stats['max_group_size']}")
    
    # Examples
    if duplicates:
        report_lines.append(f"\nEXACT DUPLICATE EXAMPLES (first 3):")
        for i, (canonical, dup_list) in enumerate(list(duplicates.items())[:3], 1):
            report_lines.append(f"  {i}. {canonical}")
            for dup in dup_list[:2]:  # Show first 2 duplicates
                report_lines.append(f"     -> {dup}")
    
    if near_duplicates:
        report_lines.append(f"\nNEAR-DUPLICATE EXAMPLES (first 3):")
        for i, (canonical, near_list) in enumerate(list(near_duplicates.items())[:3], 1):
            report_lines.append(f"  {i}. {canonical}")
            for near in near_list[:2]:  # Show first 2 near-duplicates
                report_lines.append(f"     -> {near}")
    
    report_lines.append("=" * 60)
    
    return "\n".join(report_lines)