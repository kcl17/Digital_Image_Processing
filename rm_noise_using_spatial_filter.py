import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ============================================================================
# PRACTICAL 8: SPATIAL FILTERING
# ============================================================================

# ============================================================================
# Task 8a: 1-D and 2-D Convolution Process
# ============================================================================
def convolution_demonstration():
    """
    Demonstrate 1-D and 2-D convolution process
    """
    print("Task 8a: Understanding 1-D and 2-D Convolution")
    
    # 1-D Convolution Example
    signal_1d = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
    kernel_1d = np.array([1, 2, 1]) / 4  # Smoothing kernel
    
    # Perform 1-D convolution manually
    conv_1d_manual = np.convolve(signal_1d, kernel_1d, mode='same')
    
    # 2-D Convolution Example
    # Create simple 2-D signal (small image)
    signal_2d = np.array([
        [1, 2, 3, 2, 1],
        [2, 4, 5, 4, 2],
        [3, 5, 8, 5, 3],
        [2, 4, 5, 4, 2],
        [1, 2, 3, 2, 1]
    ], dtype=np.float32)
    
    # 3x3 averaging kernel
    kernel_2d = np.ones((3, 3), dtype=np.float32) / 9
    
    # Perform 2-D convolution
    conv_2d = signal.convolve2d(signal_2d, kernel_2d, mode='same')
    
    # Display
    fig = plt.figure(figsize=(16, 10))
    
    # 1-D Convolution
    ax1 = plt.subplot(2, 3, 1)
    plt.stem(signal_1d, basefmt=' ')
    plt.title('1-D Signal', fontweight='bold', fontsize=11)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(2, 3, 2)
    plt.stem(kernel_1d, basefmt=' ', linefmt='r-', markerfmt='ro')
    plt.title('1-D Kernel [1,2,1]/4', fontweight='bold', fontsize=11)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(2, 3, 3)
    plt.stem(conv_1d_manual, basefmt=' ', linefmt='g-', markerfmt='go')
    plt.title('1-D Convolution Result\n(Smoothed Signal)', fontweight='bold', fontsize=11)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    # 2-D Convolution
    ax4 = plt.subplot(2, 3, 4)
    im1 = plt.imshow(signal_2d, cmap='viridis', interpolation='nearest')
    plt.colorbar(im1)
    plt.title('2-D Signal (5x5)', fontweight='bold', fontsize=11)
    
    ax5 = plt.subplot(2, 3, 5)
    im2 = plt.imshow(kernel_2d, cmap='hot', interpolation='nearest')
    plt.colorbar(im2)
    plt.title('2-D Kernel (3x3)\nAveraging Filter', fontweight='bold', fontsize=11)
    
    ax6 = plt.subplot(2, 3, 6)
    im3 = plt.imshow(conv_2d, cmap='viridis', interpolation='nearest')
    plt.colorbar(im3)
    plt.title('2-D Convolution Result\n(Smoothed)', fontweight='bold', fontsize=11)
    
    plt.suptitle('8a. Understanding 1-D and 2-D Convolution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=True)
    
    print("Convolution Process:")
    print("1-D: Slide kernel over signal, multiply and sum")
    print("2-D: Slide kernel over image, multiply and sum")
    print(f"Original 2D center value: {signal_2d[2,2]}")
    print(f"Convolved 2D center value: {conv_2d[2,2]:.2f}\n")
    
    return conv_1d_manual, conv_2d


# ============================================================================
# Task 8b: Low Pass and High Pass Filters with 3x3 Mask
# ============================================================================
def apply_spatial_filters(img):
    """
    Apply various spatial filters using 3x3 masks
    """
    print("Task 8b: Low Pass and High Pass Filters with 3x3 Mask")
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # ========================================================================
    # LOW PASS FILTERS (Smoothing/Blurring)
    # ========================================================================
    
    # 1. Average/Mean filter
    kernel_average = np.ones((3, 3), np.float32) / 9
    lpf_average = cv2.filter2D(gray, -1, kernel_average)
    
    # 2. Gaussian-like filter
    kernel_gaussian = np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]], np.float32) / 16
    lpf_gaussian = cv2.filter2D(gray, -1, kernel_gaussian)
    
    # 3. Weighted average
    kernel_weighted = np.array([[1, 1, 1],
                                [1, 2, 1],
                                [1, 1, 1]], np.float32) / 10
    lpf_weighted = cv2.filter2D(gray, -1, kernel_weighted)
    
    # ========================================================================
    # HIGH PASS FILTERS (Sharpening/Edge Enhancement)
    # ========================================================================
    
    # 1. Basic Laplacian (edge detection)
    kernel_laplacian = np.array([[0, -1, 0],
                                  [-1, 4, -1],
                                  [0, -1, 0]], np.float32)
    hpf_laplacian = cv2.filter2D(gray, -1, kernel_laplacian)
    hpf_laplacian = np.clip(hpf_laplacian, 0, 255).astype(np.uint8)
    
    # 2. Laplacian variant (8-neighbor)
    kernel_laplacian2 = np.array([[-1, -1, -1],
                                   [-1, 8, -1],
                                   [-1, -1, -1]], np.float32)
    hpf_laplacian2 = cv2.filter2D(gray, -1, kernel_laplacian2)
    hpf_laplacian2 = np.clip(hpf_laplacian2, 0, 255).astype(np.uint8)
    
    # 3. Sharpening filter (unsharp masking)
    kernel_sharpen = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]], np.float32)
    hpf_sharpen = cv2.filter2D(gray, -1, kernel_sharpen)
    hpf_sharpen = np.clip(hpf_sharpen, 0, 255).astype(np.uint8)
    
    # 4. High-boost filter
    kernel_highboost = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]], np.float32)
    hpf_highboost = cv2.filter2D(gray, -1, kernel_highboost)
    hpf_highboost = np.clip(hpf_highboost, 0, 255).astype(np.uint8)
    
    # Display
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Original
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original Image', fontweight='bold', fontsize=10)
    axes[0, 0].axis('off')
    
    # Low Pass Filters
    axes[0, 1].imshow(lpf_average, cmap='gray')
    axes[0, 1].set_title('LPF: Average\n[[1,1,1]/9]', fontweight='bold', fontsize=9)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(lpf_gaussian, cmap='gray')
    axes[0, 2].set_title('LPF: Gaussian-like\n[[1,2,1;2,4,2;1,2,1]/16]', 
                        fontweight='bold', fontsize=9)
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(lpf_weighted, cmap='gray')
    axes[0, 3].set_title('LPF: Weighted', fontweight='bold', fontsize=9)
    axes[0, 3].axis('off')
    
    # High Pass Filters
    axes[1, 0].imshow(hpf_laplacian, cmap='gray')
    axes[1, 0].set_title('HPF: Laplacian\n[0,-1,0;-1,4,-1;0,-1,0]', 
                        fontweight='bold', fontsize=9)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(hpf_laplacian2, cmap='gray')
    axes[1, 1].set_title('HPF: Laplacian 8-N\n[[-1,-1,-1];[-1,8,-1];[-1,-1,-1]]', 
                        fontweight='bold', fontsize=9)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(hpf_sharpen, cmap='gray')
    axes[1, 2].set_title('HPF: Sharpening\n[0,-1,0;-1,5,-1;0,-1,0]', 
                        fontweight='bold', fontsize=9)
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(hpf_highboost, cmap='gray')
    axes[1, 3].set_title('HPF: High-Boost\n[[-1,-1,-1];[-1,9,-1];[-1,-1,-1]]', 
                        fontweight='bold', fontsize=9)
    axes[1, 3].axis('off')
    
    # Kernel visualizations
    kernels = [kernel_average, kernel_gaussian, kernel_laplacian, kernel_sharpen]
    titles = ['Average Kernel', 'Gaussian Kernel', 'Laplacian Kernel', 'Sharpen Kernel']
    
    for i, (kernel, title) in enumerate(zip(kernels, titles)):
        axes[2, i].imshow(kernel, cmap='RdBu', interpolation='nearest')
        axes[2, i].set_title(title, fontweight='bold', fontsize=9)
        # Add values on the heatmap
        for (j, k), val in np.ndenumerate(kernel):
            axes[2, i].text(k, j, f'{val:.2f}', ha='center', va='center', 
                          color='white' if abs(val) > 0.3 else 'black', fontsize=8)
        axes[2, i].axis('off')
    
    plt.suptitle('8b. Spatial Filtering: Low Pass and High Pass Filters', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=True)
    
    print("\nLow Pass Filters (Smoothing):")
    print("- Reduce noise and detail")
    print("- Blur the image")
    print("- Remove high-frequency components")
    
    print("\nHigh Pass Filters (Sharpening):")
    print("- Enhance edges and details")
    print("- Amplify high-frequency components")
    print("- Detect boundaries\n")
    
    return lpf_gaussian, hpf_sharpen


# ============================================================================
# Manual Convolution Implementation
# ============================================================================
def manual_convolution_2d(img, kernel):
    """
    Implement 2-D convolution manually without using built-in functions
    """
    print("\nManual 2-D Convolution Implementation")
    
    # Get dimensions
    img_h, img_w = img.shape
    kernel_h, kernel_w = kernel.shape
    
    # Calculate padding
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    
    # Pad the image
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    # Initialize output
    output = np.zeros_like(img, dtype=np.float32)
    
    # Perform convolution
    for i in range(img_h):
        for j in range(img_w):
            # Extract region
            region = padded[i:i+kernel_h, j:j+kernel_w]
            # Element-wise multiplication and sum
            output[i, j] = np.sum(region * kernel)
    
    # Clip values
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    print("Manual convolution completed!")
    print(f"Image size: {img.shape}")
    print(f"Kernel size: {kernel.shape}")
    print(f"Output size: {output.shape}\n")
    
    return output


# ============================================================================
# Comparison: Manual vs Built-in Convolution
# ============================================================================
def compare_convolution_methods(img):
    """
    Compare manual convolution with built-in methods
    """
    print("Comparing Convolution Methods")
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Take small region for faster manual computation
    small_img = gray[100:200, 100:200]
    
    # Define kernel
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], np.float32) / 16
    
    # Method 1: Manual convolution
    import time
    start = time.time()
    manual_result = manual_convolution_2d(small_img, kernel)
    manual_time = time.time() - start
    
    # Method 2: OpenCV filter2D
    start = time.time()
    cv2_result = cv2.filter2D(small_img, -1, kernel)
    cv2_time = time.time() - start
    
    # Method 3: SciPy convolve2d
    start = time.time()
    scipy_result = signal.convolve2d(small_img, kernel, mode='same')
    scipy_result = np.clip(scipy_result, 0, 255).astype(np.uint8)
    scipy_time = time.time() - start
    
    # Calculate differences
    diff_manual_cv2 = np.abs(manual_result.astype(float) - cv2_result.astype(float))
    
    # Display
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(small_img, cmap='gray')
    axes[0, 0].set_title('Original Region', fontweight='bold', fontsize=11)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(manual_result, cmap='gray')
    axes[0, 1].set_title(f'Manual Convolution\nTime: {manual_time*1000:.2f}ms', 
                        fontweight='bold', fontsize=11)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(cv2_result, cmap='gray')
    axes[0, 2].set_title(f'OpenCV filter2D\nTime: {cv2_time*1000:.2f}ms', 
                        fontweight='bold', fontsize=11)
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(scipy_result, cmap='gray')
    axes[1, 0].set_title(f'SciPy convolve2d\nTime: {scipy_time*1000:.2f}ms', 
                        fontweight='bold', fontsize=11)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(diff_manual_cv2, cmap='hot')
    axes[1, 1].set_title(f'Difference\nMax: {diff_manual_cv2.max():.2f}', 
                        fontweight='bold', fontsize=11)
    axes[1, 1].axis('off')
    
    axes[1, 2].axis('off')
    info = f"""
    CONVOLUTION COMPARISON
    ═════════════════════
    
    Image Size: {small_img.shape}
    Kernel: 3x3 Gaussian
    
    Execution Times:
    Manual:  {manual_time*1000:.2f} ms
    OpenCV:  {cv2_time*1000:.2f} ms
    SciPy:   {scipy_time*1000:.2f} ms
    
    Speedup:
    OpenCV: {manual_time/cv2_time:.1f}x faster
    SciPy:  {manual_time/scipy_time:.1f}x faster
    
    Max Difference: {diff_manual_cv2.max():.4f}
    
    Note: Built-in functions
    use optimized algorithms
    (FFT, parallel processing)
    """
    axes[1, 2].text(0.1, 0.5, info, fontsize=9, family='monospace',
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.suptitle('Convolution Methods Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=True)
    
    print(f"Manual convolution time: {manual_time*1000:.2f} ms")
    print(f"OpenCV convolution time: {cv2_time*1000:.2f} ms")
    print(f"Speedup: {manual_time/cv2_time:.1f}x\n")


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================
def main_spatial_filtering():
    """Demonstrate spatial filtering"""
    print("="*70)
    print("PRACTICAL 8: SPATIAL FILTERING")
    print("="*70 + "\n")
    
    # Create sample image
    print("Creating sample image...\n")
    img = np.zeros((400, 400), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), 200, -1)
    cv2.circle(img, (300, 100), 50, 255, -1)
    cv2.rectangle(img, (100, 200), (350, 350), 150, -1)
    cv2.line(img, (0, 0), (400, 400), 180, 2)
    cv2.putText(img, 'FILTER', (130, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    
    cv2.imwrite('spatial_filter_sample.jpg', img)
    
    print("Press Enter to start demonstrations...")
    input()
    
    # Task 8a: Convolution demonstration
    conv_1d, conv_2d = convolution_demonstration()
    
    # Task 8b: Spatial filters
    lpf_result, hpf_result = apply_spatial_filters(img)
    
    # Bonus: Compare methods
    compare_convolution_methods(img)
    
    print("="*70)
    print("ALL SPATIAL FILTERING TASKS COMPLETED!")
    print("="*70)


if __name__ == "__main__":
    main_spatial_filtering()