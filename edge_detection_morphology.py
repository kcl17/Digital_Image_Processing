import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# PRACTICAL 10: EDGE DETECTION
# ============================================================================

def edge_detection_all_methods(img):
    """
    Apply various edge detection methods using different masks
    """
    print("PRACTICAL 10: Edge Detection Using Different Masks")
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    
    # ========================================================================
    # METHOD 1: SOBEL OPERATOR
    # ========================================================================
    # Sobel X and Y
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # Combine Sobel X and Y
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))
    
    # ========================================================================
    # METHOD 2: PREWITT OPERATOR
    # ========================================================================
    kernel_prewitt_x = np.array([[-1, 0, 1],
                                  [-1, 0, 1],
                                  [-1, 0, 1]], dtype=np.float32)
    kernel_prewitt_y = np.array([[-1, -1, -1],
                                  [0, 0, 0],
                                  [1, 1, 1]], dtype=np.float32)
    
    prewitt_x = cv2.filter2D(blurred, -1, kernel_prewitt_x)
    prewitt_y = cv2.filter2D(blurred, -1, kernel_prewitt_y)
    prewitt_combined = np.sqrt(prewitt_x**2 + prewitt_y**2).astype(np.uint8)
    
    # ========================================================================
    # METHOD 3: ROBERT'S CROSS OPERATOR
    # ========================================================================
    kernel_roberts_x = np.array([[1, 0],
                                  [0, -1]], dtype=np.float32)
    kernel_roberts_y = np.array([[0, 1],
                                  [-1, 0]], dtype=np.float32)
    
    roberts_x = cv2.filter2D(blurred, -1, kernel_roberts_x)
    roberts_y = cv2.filter2D(blurred, -1, kernel_roberts_y)
    roberts_combined = np.sqrt(roberts_x**2 + roberts_y**2).astype(np.uint8)
    
    # ========================================================================
    # METHOD 4: LAPLACIAN OPERATOR
    # ========================================================================
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    laplacian = np.uint8(np.abs(laplacian))
    
    # ========================================================================
    # METHOD 5: CANNY EDGE DETECTOR
    # ========================================================================
    canny = cv2.Canny(blurred, 50, 150)
    
    # ========================================================================
    # METHOD 6: SCHARR OPERATOR (Enhanced Sobel)
    # ========================================================================
    scharr_x = cv2.Scharr(blurred, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(blurred, cv2.CV_64F, 0, 1)
    scharr_combined = np.sqrt(scharr_x**2 + scharr_y**2)
    scharr_combined = np.uint8(np.clip(scharr_combined, 0, 255))
    
    # ========================================================================
    # METHOD 7: LOG (Laplacian of Gaussian)
    # ========================================================================
    # Apply Gaussian then Laplacian
    log = cv2.GaussianBlur(gray, (5, 5), 0)
    log = cv2.Laplacian(log, cv2.CV_64F)
    log = np.uint8(np.abs(log))
    
    # Display
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Original
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original Image', fontweight='bold', fontsize=11)
    axes[0, 0].axis('off')
    
    # Sobel
    axes[0, 1].imshow(sobel_combined, cmap='gray')
    axes[0, 1].set_title('Sobel Edge Detection\n(First Derivative)', fontweight='bold', fontsize=10)
    axes[0, 1].axis('off')
    
    # Prewitt
    axes[0, 2].imshow(prewitt_combined, cmap='gray')
    axes[0, 2].set_title('Prewitt Edge Detection\n(Similar to Sobel)', fontweight='bold', fontsize=10)
    axes[0, 2].axis('off')
    
    # Roberts
    axes[0, 3].imshow(roberts_combined, cmap='gray')
    axes[0, 3].set_title("Robert's Cross\n(2x2 mask)", fontweight='bold', fontsize=10)
    axes[0, 3].axis('off')
    
    # Laplacian
    axes[1, 0].imshow(laplacian, cmap='gray')
    axes[1, 0].set_title('Laplacian\n(Second Derivative)', fontweight='bold', fontsize=10)
    axes[1, 0].axis('off')
    
    # Canny
    axes[1, 1].imshow(canny, cmap='gray')
    axes[1, 1].set_title('Canny Edge Detector\n(Multi-stage)', fontweight='bold', fontsize=10)
    axes[1, 1].axis('off')
    
    # Scharr
    axes[1, 2].imshow(scharr_combined, cmap='gray')
    axes[1, 2].set_title('Scharr Operator\n(Enhanced Sobel)', fontweight='bold', fontsize=10)
    axes[1, 2].axis('off')
    
    # LoG
    axes[1, 3].imshow(log, cmap='gray')
    axes[1, 3].set_title('LoG (Laplacian of Gaussian)\n(Smoothed Second Derivative)', 
                        fontweight='bold', fontsize=10)
    axes[1, 3].axis('off')
    
    # Show kernels
    kernels = [
        (np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), 'Sobel X'),
        (kernel_prewitt_x, 'Prewitt X'),
        (kernel_roberts_x, 'Roberts X'),
        (np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]), 'Laplacian')
    ]
    
    for idx, (kernel, title) in enumerate(kernels):
        axes[2, idx].imshow(kernel, cmap='RdBu', interpolation='nearest')
        axes[2, idx].set_title(f'{title} Mask', fontweight='bold', fontsize=10)
        for (j, k), val in np.ndenumerate(kernel):
            axes[2, idx].text(k, j, f'{int(val)}', ha='center', va='center',
                            color='white' if abs(val) > 0.5 else 'black', fontsize=10)
        axes[2, idx].axis('off')
    
    plt.suptitle('PRACTICAL 10: Edge Detection Using Different Masks', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=True)
    
    # Comparison metrics
    print("\nEdge Detection Methods Comparison:")
    print("="*60)
    print(f"{'Method':<20} {'Non-zero pixels':<20} {'Mean intensity':<20}")
    print("="*60)
    print(f"{'Sobel':<20} {np.count_nonzero(sobel_combined):<20} {sobel_combined.mean():.2f}")
    print(f"{'Prewitt':<20} {np.count_nonzero(prewitt_combined):<20} {prewitt_combined.mean():.2f}")
    print(f"{'Roberts':<20} {np.count_nonzero(roberts_combined):<20} {roberts_combined.mean():.2f}")
    print(f"{'Laplacian':<20} {np.count_nonzero(laplacian):<20} {laplacian.mean():.2f}")
    print(f"{'Canny':<20} {np.count_nonzero(canny):<20} {canny.mean():.2f}")
    print(f"{'Scharr':<20} {np.count_nonzero(scharr_combined):<20} {scharr_combined.mean():.2f}")
    print("="*60)
    print("\nBest Methods:")
    print("- Canny: Best overall (thin, connected edges)")
    print("- Sobel/Scharr: Good for gradient-based detection")
    print("- Laplacian: Good for finding edge locations\n")
    
    return sobel_combined, canny


# ============================================================================
# PRACTICAL 11: MORPHOLOGICAL OPERATIONS
# ============================================================================

def morphological_operations(img):
    """
    Perform erosion, dilation, and other morphological operations
    """
    print("\nPRACTICAL 11: Morphological Operations - Erosion and Dilation")
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Threshold to get binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Define different structuring elements
    kernel_3x3 = np.ones((3, 3), np.uint8)
    kernel_5x5 = np.ones((5, 5), np.uint8)
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # ========================================================================
    # BASIC OPERATIONS
    # ========================================================================
    
    # 1. EROSION (makes objects smaller)
    eroded_3x3 = cv2.erode(binary, kernel_3x3, iterations=1)
    eroded_5x5 = cv2.erode(binary, kernel_5x5, iterations=1)
    eroded_2iter = cv2.erode(binary, kernel_3x3, iterations=2)
    
    # 2. DILATION (makes objects larger)
    dilated_3x3 = cv2.dilate(binary, kernel_3x3, iterations=1)
    dilated_5x5 = cv2.dilate(binary, kernel_5x5, iterations=1)
    dilated_2iter = cv2.dilate(binary, kernel_3x3, iterations=2)
    
    # ========================================================================
    # ADVANCED OPERATIONS
    # ========================================================================
    
    # 3. OPENING (erosion followed by dilation) - removes small objects
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_3x3)
    
    # 4. CLOSING (dilation followed by erosion) - fills small holes
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_3x3)
    
    # 5. GRADIENT (difference between dilation and erosion) - edge detection
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel_3x3)
    
    # 6. TOP HAT (difference between input and opening) - bright features
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_5x5)
    
    # 7. BLACK HAT (difference between closing and input) - dark features
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_5x5)
    
    # Display
    fig = plt.figure(figsize=(18, 12))
    
    # Original and binary
    plt.subplot(3, 5, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original Image', fontweight='bold', fontsize=10)
    plt.axis('off')
    
    plt.subplot(3, 5, 2)
    plt.imshow(binary, cmap='gray')
    plt.title('Binary Image', fontweight='bold', fontsize=10)
    plt.axis('off')
    
    # Erosion variants
    plt.subplot(3, 5, 3)
    plt.imshow(eroded_3x3, cmap='gray')
    plt.title('Erosion 3x3\n(1 iteration)', fontweight='bold', fontsize=10)
    plt.axis('off')
    
    plt.subplot(3, 5, 4)
    plt.imshow(eroded_5x5, cmap='gray')
    plt.title('Erosion 5x5\n(1 iteration)', fontweight='bold', fontsize=10)
    plt.axis('off')
    
    plt.subplot(3, 5, 5)
    plt.imshow(eroded_2iter, cmap='gray')
    plt.title('Erosion 3x3\n(2 iterations)', fontweight='bold', fontsize=10)
    plt.axis('off')
    
    # Dilation variants
    plt.subplot(3, 5, 6)
    plt.imshow(dilated_3x3, cmap='gray')
    plt.title('Dilation 3x3\n(1 iteration)', fontweight='bold', fontsize=10)
    plt.axis('off')
    
    plt.subplot(3, 5, 7)
    plt.imshow(dilated_5x5, cmap='gray')
    plt.title('Dilation 5x5\n(1 iteration)', fontweight='bold', fontsize=10)
    plt.axis('off')
    
    plt.subplot(3, 5, 8)
    plt.imshow(dilated_2iter, cmap='gray')
    plt.title('Dilation 3x3\n(2 iterations)', fontweight='bold', fontsize=10)
    plt.axis('off')
    
    # Advanced operations
    plt.subplot(3, 5, 9)
    plt.imshow(opening, cmap='gray')
    plt.title('Opening\n(Remove noise)', fontweight='bold', fontsize=10)
    plt.axis('off')
    
    plt.subplot(3, 5, 10)
    plt.imshow(closing, cmap='gray')
    plt.title('Closing\n(Fill holes)', fontweight='bold', fontsize=10)
    plt.axis('off')
    
    plt.subplot(3, 5, 11)
    plt.imshow(gradient, cmap='gray')
    plt.title('Morphological Gradient\n(Edge detection)', fontweight='bold', fontsize=10)
    plt.axis('off')
    
    plt.subplot(3, 5, 12)
    plt.imshow(tophat, cmap='gray')
    plt.title('Top Hat\n(Bright features)', fontweight='bold', fontsize=10)
    plt.axis('off')
    
    plt.subplot(3, 5, 13)
    plt.imshow(blackhat, cmap='gray')
    plt.title('Black Hat\n(Dark features)', fontweight='bold', fontsize=10)
    plt.axis('off')
    
    # Show structuring elements
    plt.subplot(3, 5, 14)
    plt.imshow(kernel_cross, cmap='gray', interpolation='nearest')
    plt.title('Cross Kernel', fontweight='bold', fontsize=10)
    plt.axis('off')
    
    plt.subplot(3, 5, 15)
    plt.imshow(kernel_ellipse, cmap='gray', interpolation='nearest')
    plt.title('Ellipse Kernel', fontweight='bold', fontsize=10)
    plt.axis('off')
    
    plt.suptitle('PRACTICAL 11: Morphological Operations - Erosion & Dilation', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=True)
    
    # Statistics
    print("\nMorphological Operations Statistics:")
    print("="*60)
    print(f"{'Operation':<25} {'White pixels':<20} {'% of original':<15}")
    print("="*60)
    original_white = np.count_nonzero(binary)
    print(f"{'Original':<25} {original_white:<20} {100.0:<15.1f}")
    print(f"{'Erosion 3x3':<25} {np.count_nonzero(eroded_3x3):<20} "
          f"{np.count_nonzero(eroded_3x3)/original_white*100:<15.1f}")
    print(f"{'Dilation 3x3':<25} {np.count_nonzero(dilated_3x3):<20} "
          f"{np.count_nonzero(dilated_3x3)/original_white*100:<15.1f}")
    print(f"{'Opening':<25} {np.count_nonzero(opening):<20} "
          f"{np.count_nonzero(opening)/original_white*100:<15.1f}")
    print(f"{'Closing':<25} {np.count_nonzero(closing):<20} "
          f"{np.count_nonzero(closing)/original_white*100:<15.1f}")
    print("="*60)
    print("\nKey Applications:")
    print("- Erosion: Remove small noise, shrink objects")
    print("- Dilation: Fill small holes, grow objects")
    print("- Opening: Remove small objects (noise removal)")
    print("- Closing: Fill small holes (hole filling)")
    print("- Gradient: Detect object boundaries\n")
    
    return eroded_3x3, dilated_3x3, opening, closing


# ============================================================================
# COMBINED DEMONSTRATION
# ============================================================================
def main_edge_and_morphology():
    """Demonstrate both edge detection and morphological operations"""
    print("="*70)
    print("PRACTICAL 10 & 11: EDGE DETECTION AND MORPHOLOGICAL OPERATIONS")
    print("="*70 + "\n")
    
    # Create sample image with various features
    print("Creating sample images...\n")
    
    # Image for edge detection (with gradients and edges)
    img_edge = np.zeros((400, 400), dtype=np.uint8)
    cv2.rectangle(img_edge, (50, 50), (150, 150), 200, -1)
    cv2.circle(img_edge, (300, 100), 50, 255, -1)
    cv2.rectangle(img_edge, (100, 200), (350, 350), 150, -1)
    cv2.line(img_edge, (0, 200), (400, 200), 180, 3)
    cv2.putText(img_edge, 'EDGES', (150, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    
    # Image for morphology (binary with noise)
    img_morph = np.zeros((400, 400), dtype=np.uint8)
    cv2.rectangle(img_morph, (100, 100), (300, 300), 255, -1)
    cv2.circle(img_morph, (200, 200), 70, 0, -1)  # Hole
    
    # Add salt and pepper noise
    noise_mask = np.random.random(img_morph.shape) < 0.02
    img_morph[noise_mask] = 255
    noise_mask = np.random.random(img_morph.shape) < 0.02
    img_morph[noise_mask] = 0
    
    cv2.imwrite('edge_detection_sample.jpg', img_edge)
    cv2.imwrite('morphology_sample.jpg', img_morph)
    
    print("Press Enter to start demonstrations...")
    input()
    
    # PRACTICAL 10: Edge Detection
    print("\n" + "="*70)
    sobel, canny = edge_detection_all_methods(img_edge)
    
    # PRACTICAL 11: Morphological Operations
    print("\n" + "="*70)
    eroded, dilated, opening, closing = morphological_operations(img_morph)
    
    print("\n" + "="*70)
    print("ALL PRACTICALS (10 & 11) COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nFiles saved:")
    print("- edge_detection_sample.jpg")
    print("- morphology_sample.jpg")


if __name__ == "__main__":
    main_edge_and_morphology()