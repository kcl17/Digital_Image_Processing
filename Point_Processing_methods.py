import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# PRACTICAL 2: POINT PROCESSING METHODS
# ============================================================================

# ============================================================================
# Task 2a: Obtain Negative Image
# ============================================================================
def negative_image(img):
    """
    Create negative of an image by inverting pixel values
    Formula: negative = (L-1) - original, where L=256 for 8-bit images
    """
    print("Task 2a: Negative Image")
    
    # Convert to grayscale if color
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        gray = img.copy()
        display_img = gray
    
    # Create negative: 255 - pixel_value
    negative = 255 - gray
    
    # Display
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(gray, cmap='gray')
    axes[0].set_title('Original Image', fontweight='bold', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(negative, cmap='gray')
    axes[1].set_title('Negative Image', fontweight='bold', fontsize=12)
    axes[1].axis('off')
    
    plt.suptitle('2a. Negative Image Transformation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=True)
    
    print(f"Original range: [{gray.min()}, {gray.max()}]")
    print(f"Negative range: [{negative.min()}, {negative.max()}]\n")
    
    return negative


# ============================================================================
# Task 2b: Obtain Flip Image
# ============================================================================
def flip_image(img):
    """
    Flip image horizontally, vertically, and both
    """
    print("Task 2b: Flip Image")
    
    # Convert BGR to RGB for display if color
    if len(img.shape) == 3:
        display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        display_img = img
    
    # Perform flips
    flip_horizontal = cv2.flip(img, 1)  # 1 = horizontal
    flip_vertical = cv2.flip(img, 0)    # 0 = vertical
    flip_both = cv2.flip(img, -1)       # -1 = both
    
    # Convert flipped images for display
    if len(img.shape) == 3:
        flip_h_display = cv2.cvtColor(flip_horizontal, cv2.COLOR_BGR2RGB)
        flip_v_display = cv2.cvtColor(flip_vertical, cv2.COLOR_BGR2RGB)
        flip_b_display = cv2.cvtColor(flip_both, cv2.COLOR_BGR2RGB)
        cmap = None
    else:
        flip_h_display = flip_horizontal
        flip_v_display = flip_vertical
        flip_b_display = flip_both
        cmap = 'gray'
    
    # Display
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(display_img, cmap=cmap)
    axes[0, 0].set_title('Original Image', fontweight='bold', fontsize=11)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(flip_h_display, cmap=cmap)
    axes[0, 1].set_title('Horizontal Flip (Left ↔ Right)', fontweight='bold', fontsize=11)
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(flip_v_display, cmap=cmap)
    axes[1, 0].set_title('Vertical Flip (Top ↔ Bottom)', fontweight='bold', fontsize=11)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(flip_b_display, cmap=cmap)
    axes[1, 1].set_title('Both Directions (180° rotation)', fontweight='bold', fontsize=11)
    axes[1, 1].axis('off')
    
    plt.suptitle('2b. Image Flipping Operations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=True)
    
    print("Flip operations completed:")
    print("- Horizontal flip (mirror)")
    print("- Vertical flip")
    print("- Both directions\n")
    
    return flip_horizontal, flip_vertical, flip_both


# ============================================================================
# Task 2c: Thresholding
# ============================================================================
def thresholding_operations(img):
    """
    Apply different thresholding techniques
    """
    print("Task 2c: Thresholding")
    
    # Convert to grayscale if color
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply different thresholding methods
    # 1. Binary thresholding
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # 2. Binary inverse
    _, binary_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # 3. Truncate
    _, trunc = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)
    
    # 4. To Zero
    _, tozero = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)
    
    # 5. Otsu's thresholding (automatic)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 6. Adaptive thresholding
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    
    # Display
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original', fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(binary, cmap='gray')
    axes[0, 1].set_title('Binary (T=127)', fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(binary_inv, cmap='gray')
    axes[0, 2].set_title('Binary Inverse', fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(trunc, cmap='gray')
    axes[0, 3].set_title('Truncate', fontweight='bold')
    axes[0, 3].axis('off')
    
    axes[1, 0].imshow(tozero, cmap='gray')
    axes[1, 0].set_title('To Zero', fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(otsu, cmap='gray')
    axes[1, 1].set_title("Otsu's Method", fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(adaptive, cmap='gray')
    axes[1, 2].set_title('Adaptive', fontweight='bold')
    axes[1, 2].axis('off')
    
    # Histogram
    axes[1, 3].hist(gray.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
    axes[1, 3].axvline(x=127, color='r', linestyle='--', label='Threshold=127')
    axes[1, 3].set_title('Histogram', fontweight='bold')
    axes[1, 3].set_xlabel('Pixel Intensity')
    axes[1, 3].set_ylabel('Frequency')
    axes[1, 3].legend()
    
    plt.suptitle('2c. Thresholding Techniques', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=True)
    
    print("Thresholding methods applied:")
    print("- Binary, Binary Inverse, Truncate, To Zero")
    print("- Otsu's automatic thresholding")
    print("- Adaptive thresholding\n")
    
    return binary, otsu, adaptive


# ============================================================================
# Task 2d: Contrast Stretching
# ============================================================================
def contrast_stretching(img):
    """
    Enhance image contrast using linear stretching
    Formula: new_pixel = (pixel - min) * (255 / (max - min))
    """
    print("Task 2d: Contrast Stretching")
    
    # Convert to grayscale if color
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Method 1: Linear contrast stretching (min-max normalization)
    min_val = gray.min()
    max_val = gray.max()
    stretched = ((gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    # Method 2: Percentile-based stretching (removes outliers)
    p2, p98 = np.percentile(gray, (2, 98))
    stretched_percentile = np.clip((gray - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
    
    # Method 3: Histogram equalization (non-linear)
    equalized = cv2.equalizeHist(gray)
    
    # Method 4: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    
    # Display
    fig = plt.figure(figsize=(16, 10))
    
    # Images
    ax1 = plt.subplot(2, 4, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original Image', fontweight='bold')
    plt.axis('off')
    
    ax2 = plt.subplot(2, 4, 2)
    plt.imshow(stretched, cmap='gray')
    plt.title('Linear Stretch', fontweight='bold')
    plt.axis('off')
    
    ax3 = plt.subplot(2, 4, 3)
    plt.imshow(stretched_percentile, cmap='gray')
    plt.title('Percentile Stretch', fontweight='bold')
    plt.axis('off')
    
    ax4 = plt.subplot(2, 4, 4)
    plt.imshow(equalized, cmap='gray')
    plt.title('Histogram Equalization', fontweight='bold')
    plt.axis('off')
    
    # Histograms
    ax5 = plt.subplot(2, 4, 5)
    plt.hist(gray.ravel(), bins=256, range=[0, 256], color='blue', alpha=0.7)
    plt.title(f'Original Histogram\n[{min_val}, {max_val}]', fontweight='bold')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    
    ax6 = plt.subplot(2, 4, 6)
    plt.hist(stretched.ravel(), bins=256, range=[0, 256], color='green', alpha=0.7)
    plt.title(f'Stretched Histogram\n[{stretched.min()}, {stretched.max()}]', fontweight='bold')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    
    ax7 = plt.subplot(2, 4, 7)
    plt.hist(stretched_percentile.ravel(), bins=256, range=[0, 256], color='orange', alpha=0.7)
    plt.title('Percentile Histogram', fontweight='bold')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    
    ax8 = plt.subplot(2, 4, 8)
    plt.hist(equalized.ravel(), bins=256, range=[0, 256], color='red', alpha=0.7)
    plt.title('Equalized Histogram', fontweight='bold')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    
    plt.suptitle('2d. Contrast Stretching Methods', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=True)
    
    print("Contrast stretching applied:")
    print(f"Original range: [{min_val}, {max_val}]")
    print(f"Stretched range: [{stretched.min()}, {stretched.max()}]")
    print("Methods: Linear, Percentile, Histogram Equalization, CLAHE\n")
    
    return stretched, equalized, clahe_img


# ============================================================================
# MAIN DEMONSTRATION FOR PRACTICAL 2
# ============================================================================
def main_point_processing():
    """Demonstrate all point processing operations"""
    print("="*70)
    print("PRACTICAL 2: POINT PROCESSING METHODS")
    print("="*70 + "\n")
    
    # Create sample image with varying intensities
    print("Creating sample image...\n")
    
    # Create a gradient image with patterns
    img = np.zeros((400, 400), dtype=np.uint8)
    
    # Add gradient
    for i in range(400):
        img[i, :] = int(i * 255 / 400)
    
    # Add some shapes
    cv2.circle(img, (100, 100), 50, 200, -1)
    cv2.rectangle(img, (200, 50), (350, 150), 100, -1)
    cv2.circle(img, (300, 300), 70, 255, -1)
    cv2.rectangle(img, (50, 250), (150, 350), 50, -1)
    
    # Add some noise for better demonstration
    noise = np.random.normal(0, 15, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    cv2.imwrite('sample_point_processing.jpg', img)
    
    # Execute all tasks
    print("\nPress Enter to start demonstrations...")
    input()
    
    # Task 2a: Negative
    negative = negative_image(img)
    
    # Task 2b: Flip
    flip_h, flip_v, flip_both = flip_image(img)
    
    # Task 2c: Thresholding
    binary, otsu, adaptive = thresholding_operations(img)
    
    # Task 2d: Contrast Stretching
    stretched, equalized, clahe = contrast_stretching(img)
    
    print("="*70)
    print("ALL POINT PROCESSING TASKS COMPLETED!")
    print("="*70)
    print("\nOutput files saved:")
    print("- sample_point_processing.jpg")


if __name__ == "__main__":
    main_point_processing()