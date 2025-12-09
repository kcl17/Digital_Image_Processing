import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# PRACTICAL 5: HISTOGRAM CALCULATION AND EQUALIZATION
# ============================================================================

# ============================================================================
# Task 5a: Histogram Using Inbuilt Functions
# ============================================================================
def histogram_inbuilt(img):
    """
    Calculate and display histogram using OpenCV and NumPy inbuilt functions
    """
    print("Task 5a: Histogram Using Inbuilt Functions")
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Method 1: Using OpenCV calcHist
    hist_cv = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # Method 2: Using NumPy histogram
    hist_np, bins = np.histogram(gray.ravel(), bins=256, range=[0, 256])
    
    # Method 3: Using matplotlib hist (just for comparison)
    # This will be shown in the plot directly
    
    # Calculate statistics
    mean_val = np.mean(gray)
    median_val = np.median(gray)
    std_val = np.std(gray)
    
    # Histogram equalization using inbuilt function
    equalized = cv2.equalizeHist(gray)
    hist_eq = cv2.calcHist([equalized], [0], None, [256], [0, 256])
    
    # Display
    fig = plt.figure(figsize=(16, 10))
    
    # Original image
    ax1 = plt.subplot(2, 3, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original Image', fontweight='bold', fontsize=12)
    plt.axis('off')
    
    # Histogram using OpenCV
    ax2 = plt.subplot(2, 3, 2)
    plt.plot(hist_cv, color='blue', linewidth=2)
    plt.axvline(x=mean_val, color='r', linestyle='--', label=f'Mean={mean_val:.1f}')
    plt.axvline(x=median_val, color='g', linestyle='--', label=f'Median={median_val:.1f}')
    plt.fill_between(range(256), hist_cv.flatten(), alpha=0.3)
    plt.title('Histogram (OpenCV calcHist)', fontweight='bold', fontsize=12)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 256])
    
    # Histogram using NumPy
    ax3 = plt.subplot(2, 3, 3)
    plt.bar(range(256), hist_np, color='green', alpha=0.7, width=1)
    plt.title('Histogram (NumPy histogram)', fontweight='bold', fontsize=12)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 256])
    
    # Equalized image
    ax4 = plt.subplot(2, 3, 4)
    plt.imshow(equalized, cmap='gray')
    plt.title('Equalized Image (cv2.equalizeHist)', fontweight='bold', fontsize=12)
    plt.axis('off')
    
    # Equalized histogram
    ax5 = plt.subplot(2, 3, 5)
    plt.plot(hist_eq, color='red', linewidth=2)
    plt.fill_between(range(256), hist_eq.flatten(), alpha=0.3, color='red')
    plt.title('Equalized Histogram', fontweight='bold', fontsize=12)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 256])
    
    # Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    stats_text = f"""
    HISTOGRAM STATISTICS
    ══════════════════════════
    
    Using Inbuilt Functions:
    
    OpenCV cv2.calcHist():
    - Total pixels: {int(hist_cv.sum())}
    - Max frequency: {int(hist_cv.max())}
    
    NumPy np.histogram():
    - Total pixels: {int(hist_np.sum())}
    - Max frequency: {int(hist_np.max())}
    
    Image Statistics:
    - Mean: {mean_val:.2f}
    - Median: {median_val:.2f}
    - Std Dev: {std_val:.2f}
    - Min: {gray.min()}
    - Max: {gray.max()}
    
    Shape: {gray.shape}
    Dtype: {gray.dtype}
    """
    ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.suptitle('5a. Histogram Calculation Using Inbuilt Functions', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=True)
    
    print("Inbuilt Functions Used:")
    print("- cv2.calcHist() for histogram")
    print("- np.histogram() for histogram")
    print("- cv2.equalizeHist() for equalization")
    print(f"Total pixels: {gray.size}")
    print(f"Histogram sum (OpenCV): {int(hist_cv.sum())}")
    print(f"Histogram sum (NumPy): {int(hist_np.sum())}\n")
    
    return hist_cv, equalized


# ============================================================================
# Task 5b: Histogram Without Using Inbuilt Functions
# ============================================================================
def histogram_manual(img):
    """
    Calculate histogram and perform equalization without using inbuilt functions
    Implements the algorithms from scratch
    """
    print("Task 5b: Histogram Without Using Inbuilt Functions")
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # ========================================================================
    # MANUAL HISTOGRAM CALCULATION
    # ========================================================================
    def calculate_histogram_manual(image):
        """Calculate histogram manually without inbuilt functions"""
        hist = [0] * 256  # Initialize histogram array
        
        rows, cols = image.shape
        
        # Count frequency of each intensity level
        for i in range(rows):
            for j in range(cols):
                intensity = image[i, j]
                hist[intensity] += 1
        
        return np.array(hist)
    
    # Calculate histogram manually
    hist_manual = calculate_histogram_manual(gray)
    
    # ========================================================================
    # MANUAL HISTOGRAM EQUALIZATION
    # ========================================================================
    def histogram_equalization_manual(image):
        """
        Perform histogram equalization manually
        Steps:
        1. Calculate histogram
        2. Calculate cumulative distribution function (CDF)
        3. Normalize CDF
        4. Map old intensities to new intensities
        """
        rows, cols = image.shape
        
        # Step 1: Calculate histogram
        hist = calculate_histogram_manual(image)
        
        # Step 2: Calculate CDF (Cumulative Distribution Function)
        cdf = [0] * 256
        cdf[0] = hist[0]
        for i in range(1, 256):
            cdf[i] = cdf[i-1] + hist[i]
        
        # Step 3: Normalize CDF to range [0, 255]
        cdf_min = min([val for val in cdf if val > 0])  # Find minimum non-zero CDF
        total_pixels = rows * cols
        
        # Create lookup table for transformation
        lookup_table = [0] * 256
        for i in range(256):
            lookup_table[i] = int(((cdf[i] - cdf_min) / (total_pixels - cdf_min)) * 255)
        
        # Step 4: Apply transformation
        equalized = np.zeros_like(image)
        for i in range(rows):
            for j in range(cols):
                old_intensity = image[i, j]
                equalized[i, j] = lookup_table[old_intensity]
        
        return equalized, np.array(cdf), np.array(lookup_table)
    
    # Perform manual equalization
    equalized_manual, cdf, lookup_table = histogram_equalization_manual(gray)
    hist_eq_manual = calculate_histogram_manual(equalized_manual)
    
    # ========================================================================
    # VERIFICATION WITH INBUILT FUNCTION
    # ========================================================================
    hist_cv = cv2.calcHist([gray], [0], None, [256], [0, 256])
    equalized_cv = cv2.equalizeHist(gray)
    
    # Calculate difference
    hist_diff = np.abs(hist_manual - hist_cv.flatten())
    img_diff = np.abs(equalized_manual.astype(int) - equalized_cv.astype(int))
    
    # Display
    fig = plt.figure(figsize=(18, 12))
    
    # Original image
    ax1 = plt.subplot(3, 4, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original Image', fontweight='bold', fontsize=11)
    plt.axis('off')
    
    # Manual histogram
    ax2 = plt.subplot(3, 4, 2)
    plt.bar(range(256), hist_manual, color='blue', alpha=0.7, width=1)
    plt.title('Histogram (Manual)', fontweight='bold', fontsize=11)
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.grid(True, alpha=0.3)
    
    # CDF
    ax3 = plt.subplot(3, 4, 3)
    plt.plot(cdf, color='green', linewidth=2)
    plt.title('Cumulative Distribution (CDF)', fontweight='bold', fontsize=11)
    plt.xlabel('Intensity')
    plt.ylabel('Cumulative Frequency')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 256])
    
    # Lookup table
    ax4 = plt.subplot(3, 4, 4)
    plt.plot(lookup_table, color='red', linewidth=2)
    plt.plot([0, 255], [0, 255], 'k--', alpha=0.3, label='y=x')
    plt.title('Transformation Function', fontweight='bold', fontsize=11)
    plt.xlabel('Input Intensity')
    plt.ylabel('Output Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Equalized image (manual)
    ax5 = plt.subplot(3, 4, 5)
    plt.imshow(equalized_manual, cmap='gray')
    plt.title('Equalized (Manual)', fontweight='bold', fontsize=11)
    plt.axis('off')
    
    # Equalized histogram (manual)
    ax6 = plt.subplot(3, 4, 6)
    plt.bar(range(256), hist_eq_manual, color='orange', alpha=0.7, width=1)
    plt.title('Equalized Histogram (Manual)', fontweight='bold', fontsize=11)
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.grid(True, alpha=0.3)
    
    # Equalized image (OpenCV)
    ax7 = plt.subplot(3, 4, 7)
    plt.imshow(equalized_cv, cmap='gray')
    plt.title('Equalized (OpenCV)', fontweight='bold', fontsize=11)
    plt.axis('off')
    
    # Comparison
    ax8 = plt.subplot(3, 4, 8)
    plt.imshow(img_diff, cmap='hot')
    plt.colorbar()
    plt.title(f'Difference\nMax={img_diff.max()}', fontweight='bold', fontsize=11)
    plt.axis('off')
    
    # Side-by-side histogram comparison
    ax9 = plt.subplot(3, 4, 9)
    plt.plot(hist_manual, 'b-', alpha=0.7, linewidth=2, label='Manual')
    plt.plot(hist_cv.flatten(), 'r--', alpha=0.7, linewidth=2, label='OpenCV')
    plt.title('Histogram Comparison', fontweight='bold', fontsize=11)
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 256])
    
    # Algorithm explanation
    ax10 = plt.subplot(3, 4, 10)
    ax10.axis('off')
    algo_text = """
    MANUAL ALGORITHM
    ════════════════
    
    Histogram Calculation:
    1. Initialize 256 bins
    2. For each pixel:
       - Get intensity
       - Increment bin[intensity]
    
    Equalization:
    1. Calculate histogram
    2. Compute CDF
    3. Normalize CDF to [0,255]
    4. Create lookup table
    5. Map each pixel
    
    Time: O(rows × cols)
    Space: O(256)
    """
    ax10.text(0.1, 0.5, algo_text, fontsize=9, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Statistics
    ax11 = plt.subplot(3, 4, 11)
    ax11.axis('off')
    stats_text = f"""
    VERIFICATION
    ════════════
    
    Histogram Match:
    Max diff: {hist_diff.max():.0f}
    Mean diff: {hist_diff.mean():.2f}
    
    Image Match:
    Max diff: {img_diff.max():.0f}
    Mean diff: {img_diff.mean():.2f}
    
    Original Stats:
    Mean: {gray.mean():.2f}
    Std: {gray.std():.2f}
    
    Equalized Stats:
    Mean: {equalized_manual.mean():.2f}
    Std: {equalized_manual.std():.2f}
    """
    ax11.text(0.1, 0.5, stats_text, fontsize=9, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Code snippet
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    code_text = """
    KEY CODE:
    
    # Histogram
    hist = [0] * 256
    for pixel in image:
        hist[pixel] += 1
    
    # CDF
    cdf[0] = hist[0]
    for i in range(1,256):
        cdf[i] = cdf[i-1] + hist[i]
    
    # Normalize
    lut[i] = ((cdf[i]-min)/(n-min))*255
    
    # Transform
    new[i,j] = lut[old[i,j]]
    """
    ax12.text(0.05, 0.5, code_text, fontsize=8, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    plt.suptitle('5b. Histogram and Equalization WITHOUT Inbuilt Functions', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=True)
    
    print("Manual Implementation Completed:")
    print(f"Histogram calculated manually (256 bins)")
    print(f"Equalization performed from scratch")
    print(f"Histogram match with OpenCV: max diff = {hist_diff.max():.0f}")
    print(f"Image match with OpenCV: max diff = {img_diff.max():.0f}")
    print(f"Manual implementation accuracy: {100 - (img_diff.mean()/255)*100:.2f}%\n")
    
    return hist_manual, equalized_manual, cdf, lookup_table


# ============================================================================
# MAIN DEMONSTRATION FOR PRACTICAL 5
# ============================================================================
def main_histogram_operations():
    """Demonstrate histogram operations"""
    print("="*70)
    print("PRACTICAL 5: HISTOGRAM CALCULATION AND EQUALIZATION")
    print("="*70 + "\n")
    
    # Create a low-contrast sample image
    print("Creating low-contrast sample image...\n")
    
    # Create an image with limited intensity range (low contrast)
    img = np.zeros((400, 400), dtype=np.uint8)
    
    # Add gradient with limited range
    for i in range(400):
        for j in range(400):
            img[i, j] = int(80 + (i * 60 / 400))  # Range: 80-140 (low contrast)
    
    # Add some shapes
    cv2.circle(img, (100, 100), 50, 120, -1)
    cv2.rectangle(img, (200, 50), (350, 150), 100, -1)
    cv2.circle(img, (300, 300), 70, 130, -1)
    
    # Add slight noise
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    cv2.imwrite('histogram_sample.jpg', img)
    
    print(f"Sample image created with low contrast")
    print(f"Intensity range: [{img.min()}, {img.max()}]")
    print(f"Mean: {img.mean():.2f}, Std: {img.std():.2f}\n")
    
    # Execute all tasks
    print("Press Enter to start demonstrations...")
    input()
    
    # Task 5a: Using inbuilt functions
    hist_inbuilt, eq_inbuilt = histogram_inbuilt(img)
    
    # Task 5b: Without inbuilt functions
    hist_manual, eq_manual, cdf, lut = histogram_manual(img)
    
    print("="*70)
    print("ALL HISTOGRAM OPERATIONS COMPLETED!")
    print("="*70)
    print("\nKey Learnings:")
    print("- Histogram shows distribution of pixel intensities")
    print("- Equalization spreads out intensities for better contrast")
    print("- Manual implementation matches OpenCV results")
    print("- CDF is key to histogram equalization")
    print("\nOutput files saved:")
    print("- histogram_sample.jpg")


if __name__ == "__main__":
    main_histogram_operations()