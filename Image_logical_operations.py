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
    _, binary1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
    _, binary2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY)
    or_binary = cv2.bitwise_or(binary1, binary2)
    
    # Display
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Grayscale OR
    axes[0, 0].imshow(gray1, cmap='gray')
    axes[0, 0].set_title('Image 1 (Grayscale)', fontweight='bold', fontsize=11)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gray2, cmap='gray')
    axes[0, 1].set_title('Image 2 (Grayscale)', fontweight='bold', fontsize=11)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(or_result, cmap='gray')
    axes[0, 2].set_title('OR Result\n(Grayscale)', fontweight='bold', fontsize=11)
    axes[0, 2].axis('off')
    
    # Binary OR
    axes[1, 0].imshow(binary1, cmap='gray')
    axes[1, 0].set_title('Image 1 (Binary)', fontweight='bold', fontsize=11)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(binary2, cmap='gray')
    axes[1, 1].set_title('Image 2 (Binary)', fontweight='bold', fontsize=11)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(or_binary, cmap='gray')
    axes[1, 2].set_title('OR Result\n(Binary)', fontweight='bold', fontsize=11)
    axes[1, 2].axis('off')
    
    plt.suptitle('4b. Bitwise OR Operation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=True)
    
    print("OR operation: Result = 1 if either image has 1")
    print(f"Non-zero pixels in img1: {np.count_nonzero(binary1)}")
    print(f"Non-zero pixels in img2: {np.count_nonzero(binary2)}")
    print(f"Non-zero pixels in OR: {np.count_nonzero(or_binary)}\n")
    
    return or_result, or_binary


# ============================================================================
# Task 4c: Calculate Intersection of Two Images
# ============================================================================
def intersection_of_images(img1, img2):
    """
    Calculate intersection of two images (same as AND operation)
    Shows common regions between two images
    """
    print("Task 4c: Calculate Intersection of Two Images")
    
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
    
    # Create binary masks
    _, mask1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
    _, mask2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY)
    
    # Calculate intersection (AND operation)
    intersection = cv2.bitwise_and(mask1, mask2)
    
    # Calculate union (OR operation)
    union = cv2.bitwise_or(mask1, mask2)
    
    # Calculate exclusive regions
    only_img1 = cv2.bitwise_and(mask1, cv2.bitwise_not(mask2))
    only_img2 = cv2.bitwise_and(mask2, cv2.bitwise_not(mask1))
    
    # Calculate IoU (Intersection over Union)
    intersection_area = np.count_nonzero(intersection)
    union_area = np.count_nonzero(union)
    iou = intersection_area / union_area if union_area > 0 else 0
    
    # Display
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(mask1, cmap='gray')
    axes[0, 0].set_title('Image 1 (Binary Mask)', fontweight='bold', fontsize=11)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask2, cmap='gray')
    axes[0, 1].set_title('Image 2 (Binary Mask)', fontweight='bold', fontsize=11)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(intersection, cmap='gray')
    axes[0, 2].set_title(f'Intersection\n(Common Region)', fontweight='bold', fontsize=11)
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(union, cmap='gray')
    axes[1, 0].set_title('Union\n(Combined Region)', fontweight='bold', fontsize=11)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(only_img1, cmap='Reds')
    axes[1, 1].set_title('Only in Image 1', fontweight='bold', fontsize=11)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(only_img2, cmap='Blues')
    axes[1, 2].set_title('Only in Image 2', fontweight='bold', fontsize=11)
    axes[1, 2].axis('off')
    
    plt.suptitle(f'4c. Image Intersection Analysis (IoU = {iou:.3f})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=True)
    
    print("Intersection Analysis:")
    print(f"Area in Image 1: {np.count_nonzero(mask1)} pixels")
    print(f"Area in Image 2: {np.count_nonzero(mask2)} pixels")
    print(f"Intersection area: {intersection_area} pixels")
    print(f"Union area: {union_area} pixels")
    print(f"IoU (Intersection over Union): {iou:.3f}\n")
    
    return intersection, union, iou


# ============================================================================
# Task 4d: Watermarking Using XOR Operation
# ============================================================================
def watermarking_xor(img, watermark):
    """
    Perform watermarking using XOR operation
    XOR property: A XOR B XOR B = A (reversible)
    """
    print("Task 4d: Watermarking Using XOR Operation")
    
    # Ensure watermark is same size as image
    if watermark.shape != img.shape:
        watermark = cv2.resize(watermark, (img.shape[1], img.shape[0]))
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_wm = cv2.cvtColor(watermark, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img.copy()
        gray_wm = watermark.copy()
    
    # Create a simple text watermark
    watermark_text = np.zeros_like(gray_img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(watermark_text, 'WATERMARK', (50, 200), font, 2, 255, 3)
    
    # Embedding: Original XOR Watermark = Watermarked
    watermarked = cv2.bitwise_xor(gray_img, watermark_text)
    
    # Extracting: Watermarked XOR Watermark = Original
    extracted = cv2.bitwise_xor(watermarked, watermark_text)
    
    # Verify extraction
    difference = cv2.absdiff(gray_img, extracted)
    
    # Display
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(gray_img, cmap='gray')
    axes[0, 0].set_title('1. Original Image', fontweight='bold', fontsize=11)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(watermark_text, cmap='gray')
    axes[0, 1].set_title('2. Watermark', fontweight='bold', fontsize=11)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(watermarked, cmap='gray')
    axes[0, 2].set_title('3. Watermarked Image\n(Original ⊕ Watermark)', 
                        fontweight='bold', fontsize=11)
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(extracted, cmap='gray')
    axes[1, 0].set_title('4. Extracted Image\n(Watermarked ⊕ Watermark)', 
                        fontweight='bold', fontsize=11)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(difference, cmap='hot')
    axes[1, 1].set_title('5. Difference\n|Original - Extracted|', 
                        fontweight='bold', fontsize=11)
    axes[1, 1].axis('off')
    
    # Process flow diagram
    axes[1, 2].axis('off')
    flow_text = """
    XOR WATERMARKING PROCESS
    ═══════════════════════
    
    EMBEDDING:
    Original ⊕ Watermark 
         ↓
    Watermarked Image
    
    EXTRACTION:
    Watermarked ⊕ Watermark
         ↓
    Original Image
    
    PROPERTY:
    A ⊕ B ⊕ B = A
    (Reversible!)
    
    ⊕ = XOR operation
    """
    axes[1, 2].text(0.1, 0.5, flow_text, fontsize=10, family='monospace',
                   verticalalignment='center', 
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.suptitle('4d. XOR-Based Watermarking', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=True)
    
    perfect_recovery = np.array_equal(gray_img, extracted)
    print("XOR Watermarking Analysis:")
    print(f"Original image mean: {gray_img.mean():.2f}")
    print(f"Watermarked image mean: {watermarked.mean():.2f}")
    print(f"Extracted image mean: {extracted.mean():.2f}")
    print(f"Perfect recovery: {perfect_recovery}")
    print(f"Maximum difference: {difference.max()}\n")
    
    return watermarked, extracted, watermark_text


# ============================================================================
# Task 4e: NOT Operation (Negative Image)
# ============================================================================
def not_operation(img):
    """
    Perform bitwise NOT operation (complement)
    Creates negative/inverted image
    """
    print("Task 4e: NOT Operation (Negative Image)")
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Perform NOT operation (bitwise complement)
    not_result = cv2.bitwise_not(gray)
    
    # Alternative: 255 - gray
    inverted = 255 - gray
    
    # Create binary version
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    not_binary = cv2.bitwise_not(binary)
    
    # Verify both methods give same result
    same_result = np.array_equal(not_result, inverted)
    
    # Display
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original Image\n(Grayscale)', fontweight='bold', fontsize=11)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(not_result, cmap='gray')
    axes[0, 1].set_title('NOT Result\n(bitwise_not)', fontweight='bold', fontsize=11)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(inverted, cmap='gray')
    axes[0, 2].set_title('Inverted\n(255 - pixel)', fontweight='bold', fontsize=11)
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(binary, cmap='gray')
    axes[1, 0].set_title('Binary Original', fontweight='bold', fontsize=11)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(not_binary, cmap='gray')
    axes[1, 1].set_title('Binary Inverted', fontweight='bold', fontsize=11)
    axes[1, 1].axis('off')
    
    # Histogram comparison
    axes[1, 2].hist(gray.ravel(), bins=256, range=[0, 256], 
                   color='blue', alpha=0.5, label='Original')
    axes[1, 2].hist(not_result.ravel(), bins=256, range=[0, 256], 
                   color='red', alpha=0.5, label='Inverted')
    axes[1, 2].set_title('Histogram Comparison', fontweight='bold', fontsize=11)
    axes[1, 2].set_xlabel('Pixel Intensity')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('4e. Bitwise NOT Operation (Negative Image)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=True)
    
    print("NOT operation: Inverts all bits")
    print(f"Original range: [{gray.min()}, {gray.max()}]")
    print(f"NOT result range: [{not_result.min()}, {not_result.max()}]")
    print(f"Both methods produce same result: {same_result}\n")
    
    return not_result


# ============================================================================
# MAIN DEMONSTRATION FOR PRACTICAL 4
# ============================================================================
def main_logical_operations():
    """Demonstrate all logical operations"""
    print("="*70)
    print("PRACTICAL 4: IMAGE LOGICAL OPERATIONS")
    print("="*70 + "\n")
    
    # Create sample images with clear patterns
    print("Creating sample images...\n")
    
    # Image 1: Circle
    img1 = np.zeros((400, 400), dtype=np.uint8)
    cv2.circle(img1, (200, 200), 150, 255, -1)
    
    # Image 2: Rectangle
    img2 = np.zeros((400, 400), dtype=np.uint8)
    cv2.rectangle(img2, (100, 100), (300, 300), 255, -1)
    
    # Image 3: Gradient for watermarking
    img3 = np.zeros((400, 400), dtype=np.uint8)
    for i in range(400):
        img3[i, :] = int(i * 255 / 400)
    cv2.circle(img3, (200, 200), 100, 200, -1)
    
    cv2.imwrite('logical_img1.jpg', img1)
    cv2.imwrite('logical_img2.jpg', img2)
    cv2.imwrite('logical_img3.jpg', img3)
    
    # Execute all tasks
    print("Press Enter to start demonstrations...")
    input()
    
    # Task 4a: AND
    and_result, and_binary = and_operation(img1, img2)
    
    # Task 4b: OR
    or_result, or_binary = or_operation(img1, img2)
    
    # Task 4c: Intersection
    intersection, union, iou = intersection_of_images(img1, img2)
    
    # Task 4d: XOR Watermarking
    watermarked, extracted, watermark = watermarking_xor(img3, img1)
    
    # Task 4e: NOT
    not_result = not_operation(img3)
    
    print("="*70)
    print("ALL LOGICAL OPERATIONS COMPLETED!")
    print("="*70)
    print("\nOutput files saved:")
    print("- logical_img1.jpg")
    print("- logical_img2.jpg")
    print("- logical_img3.jpg")


if __name__ == "__main__":
    main_logical_operations()