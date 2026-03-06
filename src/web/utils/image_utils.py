"""
Image utility functions for encoding/decoding
"""

import cv2
import numpy as np
import base64
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)


def decode_base64_image(base64_string: str) -> np.ndarray:
    """
    Decode base64 string to OpenCV image
    
    Args:
        base64_string: Base64 encoded image string (with or without data URI prefix)
        
    Returns:
        OpenCV image as numpy array
        
    Raises:
        HTTPException: If image decoding fails
    """
    try:
        # Remove data URI header if present (e.g., "data:image/jpeg;base64,")
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64 to bytes
        img_data = base64.b64decode(base64_string)
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image data")
        
        return img
        
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid image data: {str(e)}"
        )


def encode_image_to_base64(image: np.ndarray, quality: int = 85) -> str:
    """
    Encode OpenCV image to base64 string
    
    Args:
        image: OpenCV image as numpy array
        quality: JPEG quality (1-100, default 85)
        
    Returns:
        Base64 encoded string
    """
    try:
        # Encode image to JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, buffer = cv2.imencode('.jpg', image, encode_params)
        
        # Convert to base64
        return base64.b64encode(buffer).decode('utf-8')
        
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        raise


def validate_image(image: np.ndarray) -> bool:
    """
    Validate that image is a valid OpenCV image
    
    Args:
        image: Image to validate
        
    Returns:
        True if valid, False otherwise
    """
    if image is None:
        return False
    
    if not isinstance(image, np.ndarray):
        return False
    
    if len(image.shape) not in (2, 3):
        return False
    
    return True


def resize_image(image: np.ndarray, max_width: int = 1280, max_height: int = 720) -> np.ndarray:
    """
    Resize image if it exceeds maximum dimensions while maintaining aspect ratio
    
    Args:
        image: Original image
        max_width: Maximum width
        max_height: Maximum height
        
    Returns:
        Resized image
    """
    height, width = image.shape[:2]
    
    if width <= max_width and height <= max_height:
        return image
    
    # Calculate scaling factor
    scale = min(max_width / width, max_height / height)
    
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
