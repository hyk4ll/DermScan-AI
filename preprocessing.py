# -*- coding: utf-8 -*-
"""
Skin Disease Detection System - Preprocessing Module
Author: Muhammad Haikhal Bin Omanudin Baki
Project: Skin Disease Detection System Using Deep Learning (FYP)

This module contains preprocessing functions for skin disease images,
including hair removal using morphological operations and inpainting.
"""

import cv2
import numpy as np
from PIL import Image


def hair_removal(image, return_mask=False):
    """
    Remove hair from dermoscopic images using morphological black-hat transform
    and inpainting technique.
    
    Methodology (Section 3.5.1):
    1. Convert to grayscale
    2. Apply morphological black-hat transform to detect dark hair structures
    3. Create binary mask using thresholding
    4. Apply inpainting to fill masked regions with surrounding pixels
    
    Args:
        image: Input image (numpy array in RGB format or file path string)
        return_mask: If True, returns the hair mask along with processed image
        
    Returns:
        Processed image with hair removed (RGB format)
        If return_mask=True, returns tuple: (processed_image, mask)
    """
    # Handle file path input
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Could not read image from path: {image}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image.copy()
    
    # Ensure image is uint8
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Morphological Black Hat transform
    # Kernel size 9x9 targets thin hair-like structures without
    # accidentally masking lesion texture or fine clinical detail.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Thresholding to create a binary mask
    # Threshold 15 filters out low-contrast noise from the blackhat,
    # keeping only prominent hair-like edges.
    _, mask = cv2.threshold(blackhat, 15, 255, cv2.THRESH_BINARY)
    
    # Inpainting using Telea algorithm
    # Radius 6 gives smoother fill for real hair strands while the
    # tighter mask prevents unnecessary blurring of lesion areas.
    result = cv2.inpaint(img, mask, inpaintRadius=6, flags=cv2.INPAINT_TELEA)
    
    if return_mask:
        return result, mask
    return result


def preprocess_for_prediction(image, target_size=(128, 128)):
    """
    Preprocess a single image for model prediction.
    
    Steps:
    1. Apply hair removal
    2. Resize to target size
    3. Normalize pixel values to [0, 1]
    
    Args:
        image: Input image (numpy array RGB or PIL Image or file path)
        target_size: Tuple of (height, width) for resizing
        
    Returns:
        Preprocessed image ready for model prediction (shape: 1 x H x W x 3)
    """
    # Convert PIL Image to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Apply hair removal
    processed = hair_removal(image)
    
    # Resize
    processed = cv2.resize(processed, target_size)
    
    # Normalize to [0, 1]
    processed = processed.astype(np.float32) / 255.0
    
    # Add batch dimension
    processed = np.expand_dims(processed, axis=0)
    
    return processed


def custom_preprocessing_generator(img):
    """
    Custom preprocessing function for Keras ImageDataGenerator.
    Applies hair removal to each image during training/validation.
    
    Args:
        img: Input image from generator (numpy array)
        
    Returns:
        Processed image with hair removed (float32)
    """
    # Convert to uint8 for OpenCV operations
    img_uint8 = img.astype(np.uint8)
    
    # Apply hair removal
    result = hair_removal(img_uint8)
    
    # Return as float32 (generator will handle rescaling)
    return result.astype(np.float32)


def visualize_preprocessing(image_path, save_path=None):
    """
    Visualize the hair removal preprocessing pipeline.
    Shows original image, detected hair mask, and cleaned image.
    
    Args:
        image_path: Path to the input image
        save_path: Optional path to save the visualization
    """
    import matplotlib.pyplot as plt
    
    # Read original image
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Apply hair removal and get mask
    cleaned, mask = hair_removal(original, return_mask=True)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Hair Mask (BlackHat + Threshold)")
    axes[1].axis('off')
    
    axes[2].imshow(cleaned)
    axes[2].set_title("Inpainted (Hair Removed)")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


# Disease class mapping
LESION_TYPE_DICT = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma',
    'healthy': 'Healthy skin'
}

# Disease descriptions for the web interface
DISEASE_INFO = {
    'Melanocytic nevi': {
        'description': 'Commonly known as moles. Benign neoplasms of melanocytes.',
        'severity': 'Low',
        'recommendation': 'Generally harmless. Monitor for changes in size, shape, or color.'
    },
    'Melanoma': {
        'description': 'A serious form of skin cancer that develops from melanocytes.',
        'severity': 'High',
        'recommendation': 'URGENT: Please consult a dermatologist immediately for proper diagnosis.'
    },
    'Benign keratosis-like lesions': {
        'description': 'Includes seborrheic keratoses, solar lentigo, and lichen-planus like keratoses.',
        'severity': 'Low',
        'recommendation': 'Usually harmless. Consult if concerned about appearance or growth.'
    },
    'Basal cell carcinoma': {
        'description': 'Most common type of skin cancer. Slow-growing and rarely spreads.',
        'severity': 'Medium-High',
        'recommendation': 'Please consult a dermatologist for proper evaluation and treatment.'
    },
    'Actinic keratoses': {
        'description': 'Rough, scaly patches caused by sun damage. Pre-cancerous condition.',
        'severity': 'Medium',
        'recommendation': 'Consult a dermatologist. Can develop into squamous cell carcinoma if untreated.'
    },
    'Vascular lesions': {
        'description': 'Includes angiomas, angiokeratomas, pyogenic granulomas, and hemorrhage.',
        'severity': 'Low',
        'recommendation': 'Usually benign. Consult if bleeding or rapidly changing.'
    },
    'Dermatofibroma': {
        'description': 'Common benign skin growth, usually firm and slightly raised.',
        'severity': 'Low',
        'recommendation': 'Harmless growths. Removal is optional and usually for cosmetic reasons.'
    },
    'Healthy skin': {
        'description': 'No visible skin lesion or abnormality detected. The skin appears normal and healthy.',
        'severity': 'None',
        'recommendation': 'No action needed. Your skin looks healthy! Continue regular skin checks and sun protection.'
    }
}


if __name__ == "__main__":
    # Example usage
    import os
    from glob import glob
    
    # Find a sample image
    image_paths = glob(os.path.join("HAM10000_images_part_1", "*.jpg"))
    if image_paths:
        print("Testing preprocessing on sample image...")
        visualize_preprocessing(image_paths[0])
        print("Preprocessing test completed!")
    else:
        print("No images found in HAM10000_images_part_1/")
