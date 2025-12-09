
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener

# ============================================================================
# PRACTICAL 7: NOISE MODELS AND IMAGE RESTORATION
# ============================================================================

# ============================================================================
# Helper: Add Different Types of Noise
# ============================================================================
def add_gaussian_noise(img, mean=0, sigma=25):
    """Add Gaussian noise to image"""
    noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy

def add_salt_pepper_noise(img, salt_prob=0.01, pepper_prob=0.01):
    """Add salt and pepper noise to image"""
    noisy = img.copy()
    
    # Salt noise (white pixels)
    salt_mask = np.random.random(img.shape) < salt_prob
    noisy[salt_mask] = 255
    
    # Pepper noise (black pixels)
    pepper_mask = np.random.random(img.shape) < pepper_prob
    noisy[pepper_mask] = 0
    
    return noisy

def add_speckle_noise(img, noise_level=0.1):
    """Add speckle (multiplicative) noise"""
    noise = np.random.randn(*img.shape) * noise_level
    noisy = img + img * noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_poisson_noise(img):
    """Add Poisson noise"""
    noisy = np.random.poisson(img).astype(np.uint8)
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ============================================================================
# Task 7a: Image Restoration
# ============================================================================
def image_restoration_demo(img):
    """
    Demonstrate various image restoration techniques
    """
    print("Task 7a: Image Restoration")
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Add different types of noise
    noisy_gaussian = add_gaussian_noise(gray, mean=0, sigma=20)
    noisy_sp = add_salt_pepper_noise(gray, salt_prob=0.02, pepper_prob=0.02)
    noisy_speckle = add_speckle_noise(gray, noise_level=0.15)
    
    # Apply restoration techniques
    # 1. Mean filter (good for Gaussian noise)
    restored_mean = cv2.blur(noisy_gaussian, (5, 5))
    
    # 2. Median filter (excellent for salt & pepper)
    restored_median = cv2.medianBlur(noisy_sp, 5)
    
    # 3. Gaussian filter
    restored_gaussian = cv2.GaussianBlur(noisy_gaussian, (5, 5), 1.5)
    
    # 4. Bilateral filter (edge-preserving)
    restored_bilateral = cv2.bilateralFilter(noisy_gaussian, 9, 75, 75)
    
    # 5. Non-local means denoising
    restored_nlm = cv2.fastNlMeansDenoising(noisy_gaussian, h=10)
    
    # Display
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original Clean', fontweight='bold', fontsize=10)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(noisy_gaussian, cmap='gray')
    axes[0, 1].set_title('Gaussian Noise Added', fontweight='bold', fontsize=10)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(restored_mean, cmap='gray')
    axes[0, 2].set_title('Mean Filter Restored', fontweight='bold', fontsize=10)
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(restored_gaussian, cmap='gray')
    axes[1, 0].set_title('Gaussian Filter', fontweight='bold', fontsize=10)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(restored_bilateral, cmap='gray')
    axes[1, 1].set_title('Bilateral Filter\n(Edge-Preserving)', fontweight='bold', fontsize=10)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(restored_nlm, cmap='gray')
    axes[1, 2].set_title('Non-Local Means', fontweight='bold', fontsize=10)
    axes[1, 2].axis('off')
    
    axes[2, 0].imshow(noisy_sp, cmap='gray')
    axes[2, 0].set_title('Salt & Pepper Noise', fontweight='bold', fontsize=10)
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(restored_median, cmap='gray')
    axes[2, 1].set_title('Median Filter\n(Best for S&P)', fontweight='bold', fontsize=10)
    axes[2, 1].axis('off')
    
    # Calculate PSNR
    psnr_median = cv2.PSNR(gray, restored_median)
    
    axes[2, 2].axis('off')
    info = f"""
    RESTORATION METHODS
    ══════════════════
    
    1. Mean Filter
       - Average of neighbors
       - Good for Gaussian
    
    2. Median Filter
       - Middle value
       - Best for S&P
    
    3. Gaussian Filter
       - Weighted average
       - Smooth results
    
    4. Bilateral Filter
       - Edge-preserving
       - Smart smoothing
    
    5. Non-Local Means
       - Pattern matching
       - High quality
    
    PSNR (Median): {psnr_median:.2f} dB
    """
    axes[2, 2].text(0.1, 0.5, info, fontsize=8, family='monospace',
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.suptitle('7a. Image Restoration Techniques', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=True)
    
    print("Restoration techniques applied:")
    print("- Mean, Gaussian, Median filters")
    print("- Bilateral (edge-preserving)")
    print("- Non-local means denoising\n")
    
    return restored_median, restored_bilateral


# ============================================================================
# Task 7b: Remove Salt and Pepper Noise
# ============================================================================
def remove_salt_pepper(img, noise_amount=0.05):
    """
    Remove salt and pepper noise using various filters
    """
    print("Task 7b: Remove Salt and Pepper Noise")
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Add salt and pepper noise
    noisy = add_salt_pepper_noise(gray, salt_prob=noise_amount, pepper_prob=noise_amount)
    
    # Method 1: Median filter (most effective)
    restored_median_3 = cv2.medianBlur(noisy, 3)
    restored_median_5 = cv2.medianBlur(noisy, 5)
    restored_median_7 = cv2.medianBlur(noisy, 7)
    
    # Method 2: Mean filter (less effective)
    restored_mean = cv2.blur(noisy, (5, 5))
    
    # Method 3: Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    restored_morph_open = cv2.morphologyEx(noisy, cv2.MORPH_OPEN, kernel)
    restored_morph_close = cv2.morphologyEx(noisy, cv2.MORPH_CLOSE, kernel)
    
    # Calculate metrics
    psnr_median3 = cv2.PSNR(gray, restored_median_3)
    psnr_median5 = cv2.PSNR(gray, restored_median_5)
    psnr_median7 = cv2.PSNR(gray, restored_median_7)
    
    # Display
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original', fontweight='bold', fontsize=10)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(noisy, cmap='gray')
    axes[0, 1].set_title(f'Salt & Pepper\n({noise_amount*100:.0f}% noise)', 
                        fontweight='bold', fontsize=10)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(restored_median_3, cmap='gray')
    axes[0, 2].set_title(f'Median 3x3\nPSNR={psnr_median3:.1f}', 
                        fontweight='bold', fontsize=10)
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(restored_median_5, cmap='gray')
    axes[0, 3].set_title(f'Median 5x5\nPSNR={psnr_median5:.1f}', 
                        fontweight='bold', fontsize=10)
    axes[0, 3].axis('off')
    
    axes[1, 0].imshow(restored_median_7, cmap='gray')
    axes[1, 0].set_title(f'Median 7x7\nPSNR={psnr_median7:.1f}', 
                        fontweight='bold', fontsize=10)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(restored_mean, cmap='gray')
    axes[1, 1].set_title('Mean Filter\n(Less Effective)', fontweight='bold', fontsize=10)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(restored_morph_open, cmap='gray')
    axes[1, 2].set_title('Morphological Open\n(Removes Salt)', 
                        fontweight='bold', fontsize=10)
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(restored_morph_close, cmap='gray')
    axes[1, 3].set_title('Morphological Close\n(Removes Pepper)', 
                        fontweight='bold', fontsize=10)
    axes[1, 3].axis('off')
    
    plt.suptitle('7b. Salt & Pepper Noise Removal', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=True)
    
    print("Salt & Pepper noise removal:")
    print(f"Noise amount: {noise_amount*100:.0f}%")
    print(f"Best method: Median filter 5x5 (PSNR={psnr_median5:.2f} dB)")
    print("Median filter is most effective for impulse noise\n")
    
    return restored_median_5


# ============================================================================
# Task 7c: Minimize Gaussian Noise
# ============================================================================
def minimize_gaussian_noise(img, sigma=25):
    """
    Minimize Gaussian noise using various filters
    """
    print("Task 7c: Minimize Gaussian Noise")
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Add Gaussian noise
    noisy = add_gaussian_noise(gray, mean=0, sigma=sigma)
    
    # Method 1: Mean filter
    restored_mean = cv2.blur(noisy, (5, 5))
    
    # Method 2: Gaussian filter (different sigmas)
    restored_gauss_small = cv2.GaussianBlur(noisy, (5, 5), 1.0)
    restored_gauss_med = cv2.GaussianBlur(noisy, (7, 7), 1.5)
    restored_gauss_large = cv2.GaussianBlur(noisy, (9, 9), 2.0)
    
    # Method 3: Bilateral filter (edge-preserving)
    restored_bilateral = cv2.bilateralFilter(noisy, 9, 75, 75)
    
    # Method 4: Non-local means
    restored_nlm = cv2.fastNlMeansDenoising(noisy, h=10, templateWindowSize=7, 
                                            searchWindowSize=21)
    
    # Calculate PSNR
    psnr_mean = cv2.PSNR(gray, restored_mean)
    psnr_gauss = cv2.PSNR(gray, restored_gauss_med)
    psnr_bilateral = cv2.PSNR(gray, restored_bilateral)
    psnr_nlm = cv2.PSNR(gray, restored_nlm)
    
    # Display
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original', fontweight='bold', fontsize=10)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(noisy, cmap='gray')
    axes[0, 1].set_title(f'Gaussian Noise\n(σ={sigma})', fontweight='bold', fontsize=10)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(restored_mean, cmap='gray')
    axes[0, 2].set_title(f'Mean Filter\nPSNR={psnr_mean:.1f}', 
                        fontweight='bold', fontsize=10)
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(restored_gauss_small, cmap='gray')
    axes[0, 3].set_title('Gaussian σ=1.0', fontweight='bold', fontsize=10)
    axes[0, 3].axis('off')
    
    axes[1, 0].imshow(restored_gauss_med, cmap='gray')
    axes[1, 0].set_title(f'Gaussian σ=1.5\nPSNR={psnr_gauss:.1f}', 
                        fontweight='bold', fontsize=10)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(restored_gauss_large, cmap='gray')
    axes[1, 1].set_title('Gaussian σ=2.0', fontweight='bold', fontsize=10)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(restored_bilateral, cmap='gray')
    axes[1, 2].set_title(f'Bilateral\nPSNR={psnr_bilateral:.1f}', 
                        fontweight='bold', fontsize=10)
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(restored_nlm, cmap='gray')
    axes[1, 3].set_title(f'Non-Local Means\nPSNR={psnr_nlm:.1f}', 
                        fontweight='bold', fontsize=10)
    axes[1, 3].axis('off')
    
    plt.suptitle('7c. Gaussian Noise Minimization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=True)
    
    print("Gaussian noise minimization:")
    print(f"Noise sigma: {sigma}")
    print(f"Mean filter PSNR: {psnr_mean:.2f} dB")
    print(f"Gaussian filter PSNR: {psnr_gauss:.2f} dB")
    print(f"Bilateral filter PSNR: {psnr_bilateral:.2f} dB")
    print(f"Non-local means PSNR: {psnr_nlm:.2f} dB (Best!)\n")
    
    return restored_nlm


# ============================================================================
# Task 7d: Median Filter and Wiener Filter
# ============================================================================
def median_and_wiener_filter(img):
    """
    Apply Median and Wiener filters for noise removal
    """
    print("Task 7d: Median Filter and Wiener Filter")
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Add different noises
    noisy_sp = add_salt_pepper_noise(gray, salt_prob=0.03, pepper_prob=0.03)
    noisy_gauss = add_gaussian_noise(gray, mean=0, sigma=20)
    
    # Median filter (different kernel sizes)
    median_3 = cv2.medianBlur(noisy_sp, 3)
    median_5 = cv2.medianBlur(noisy_sp, 5)
    median_7 = cv2.medianBlur(noisy_sp, 7)
    
    # Wiener filter (using scipy)
    # Estimate noise from noisy image
    wiener_result = wiener(noisy_gauss, (5, 5))
    wiener_result = np.clip(wiener_result, 0, 255).astype(np.uint8)
    
    # Display
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original', fontweight='bold', fontsize=10)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(noisy_sp, cmap='gray')
    axes[0, 1].set_title('Salt & Pepper Noise', fontweight='bold', fontsize=10)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(median_3, cmap='gray')
    axes[0, 2].set_title('Median Filter 3x3', fontweight='bold', fontsize=10)
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(median_5, cmap='gray')
    axes[0, 3].set_title('Median Filter 5x5', fontweight='bold', fontsize=10)
    axes[0, 3].axis('off')
    
    axes[1, 0].imshow(median_7, cmap='gray')
    axes[1, 0].set_title('Median Filter 7x7', fontweight='bold', fontsize=10)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(noisy_gauss, cmap='gray')
    axes[1, 1].set_title('Gaussian Noise', fontweight='bold', fontsize=10)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(wiener_result, cmap='gray')
    axes[1, 2].set_title('Wiener Filter Result', fontweight='bold', fontsize=10)
    axes[1, 2].axis('off')
    
    axes[1, 3].axis('off')
    info = """
    FILTER COMPARISON
    ═════════════════
    
    MEDIAN FILTER:
    - Non-linear filter
    - Replaces pixel with
      median of neighbors
    - Best for: Salt & Pepper
    - Preserves edges
    - Simple, effective
    
    WIENER FILTER:
    - Linear filter
    - Inverse filtering
    - Noise estimation
    - Best for: Gaussian
    - Optimal MSE
    - More complex
    
    Use Case:
    Median → Impulse noise
    Wiener → Gaussian noise
    """
    axes[1, 3].text(0.05, 0.5, info, fontsize=8, family='monospace',
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    plt.suptitle('7d. Median Filter vs Wiener Filter', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=True)
    
    print("Filter comparison:")
    print("- Median filter: Best for salt & pepper (impulse) noise")
    print("- Wiener filter: Best for Gaussian noise with known characteristics")
    print("- Median preserves edges better")
    print("- Wiener provides optimal MSE reduction\n")
    
    return median_5, wiener_result


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================
def main_noise_restoration():
    """Demonstrate noise models and restoration"""
    print("="*70)
    print("PRACTICAL 7: NOISE MODELS AND IMAGE RESTORATION")
    print("="*70 + "\n")
    
    # Create sample image
    print("Creating sample image...\n")
    img = np.zeros((400, 400), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), 200, -1)
    cv2.circle(img, (300, 100), 50, 255, -1)
    cv2.rectangle(img, (100, 200), (350, 350), 150, -1)
    cv2.putText(img, 'DENOISE', (120, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    
    cv2.imwrite('noise_sample.jpg', img)
    
    print("Press Enter to start demonstrations...")
    input()
    
    # Execute all tasks
    restored_med, restored_bil = image_restoration_demo(img)
    restored_sp = remove_salt_pepper(img, noise_amount=0.05)
    restored_gauss = minimize_gaussian_noise(img, sigma=25)
    median_result, wiener_result = median_and_wiener_filter(img)
    
    print("="*70)
    print("ALL NOISE RESTORATION TASKS COMPLETED!")
    print("="*70)


if __name__ == "__main__":
    main_noise_restoration()