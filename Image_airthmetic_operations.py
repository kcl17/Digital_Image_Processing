import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# PRACTICAL 4: IMAGE LOGICAL OPERATIONS
# ============================================================================

# ============================================================================
# Task 4a: AND Operation Between Two Images
# ============================================================================
def and_operation(img1, img2):
    """
    Perform bitwise AND operation between two images
    Result pixel is 1 only if both input pixels are 1
    """
    print("Task 4a: AND Operation Between Two Images")
    
    # Ensure images are same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1.copy()
        gray2 = img2.copy()
    
    # Perform AND operation
    and_result = cv2.bitwise_and(gray1, gray2)
    
    # Also show binary version for clearer understanding
    _, binary1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
    _, binary2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY)
    and_binary = cv2.bitwise_and(binary1, binary2)
    
    # Display
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Grayscale AND
    axes[0, 0].imshow(gray1, cmap='gray')
    axes[0, 0].set_title('Image 1 (Grayscale)', fontweight='bold', fontsize=11)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gray2, cmap='gray')
    axes[0, 1].set_title('Image 2 (Grayscale)', fontweight='bold', fontsize=11)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(and_result, cmap='gray')
    axes[0, 2].set_title('AND Result\n(Grayscale)', fontweight='bold', fontsize=11)
    axes[0, 2].axis('off')
    
    # Binary AND
    axes[1, 0].imshow(binary1, cmap='gray')
    axes[1, 0].set_title('Image 1 (Binary)', fontweight='bold', fontsize=11)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(binary2, cmap='gray')
    axes[1, 1].set_title('Image 2 (Binary)', fontweight='bold', fontsize=11)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(and_binary, cmap='gray')
    axes[1, 2].set_title('AND Result\n(Binary)', fontweight='bold', fontsize=11)
    axes[1, 2].axis('off')
    
    plt.suptitle('4a. Bitwise AND Operation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=True)
    
    print("AND operation: Result = 1 only where both images have 1")
    print(f"Non-zero pixels in img1: {np.count_nonzero(binary1)}")
    print(f"Non-zero pixels in img2: {np.count_nonzero(binary2)}")
    print(f"Non-zero pixels in AND: {np.count_nonzero(and_binary)}\n")
    
    return and_result, and_binary


# ============================================================================
# Task 4b: OR Operation Between Two Images
# ============================================================================
def or_operation(img1, img2):
    """
    Perform bitwise OR operation between two images
    Result pixel is 1 if either input pixel is 1
    """
    print("Task 4b: OR Operation Between Two Images")
    
    # Ensure images are same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1.copy()
        gray2 = img2.copy()
    
    # Perform OR operation
    or_result = cv2.bitwise_or(gray1, gray2)
    
    # Binary version
    _, binary1 = cv