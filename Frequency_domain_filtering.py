import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# PRACTICAL 9: FREQUENCY DOMAIN FILTERING
# ============================================================================

# ============================================================================
# Task 9a: Apply FFT on Given Image
# ============================================================================
def apply_fft(img):
    """
    Apply Fast Fourier Transform (FFT) on image
    """
    print("Task 9a: Apply FFT on Given Image")
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply FFT
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)  # Shift zero frequency to center
    
    # Calculate magnitude and phase
    magnitude_spectrum = np.abs(fft_shift)
    phase_spectrum = np.angle(fft_shift)
    
    # Log transform for better visualization
    magnitude_spectrum_log = np.log1p(magnitude_spectrum)
    
    # Display
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original Image', fontweight='bold', fontsize=11)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(magnitude_spectrum, cmap='hot')
    axes[0, 1].set_title('Magnitude Spectrum\n(Linear Scale)', fontweight='bold', fontsize=11)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(magnitude_spectrum_log, cmap='hot')
    axes[0, 2].set_title('Magnitude Spectrum\n(Log Scale)', fontweight='bold', fontsize=11)
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(phase_spectrum, cmap='hsv')
    axes[1, 0].set_title('Phase Spectrum', fontweight='bold', fontsize=11)
    axes[1, 0].axis('off')
    
    # Show frequency components
    axes[1, 1].imshow(np.abs(fft[:gray.shape[0]//2, :gray.shape[1]//2]), cmap='hot')
    axes[1, 1].set_title('Top-Left Quadrant\n(Low Frequencies)', fontweight='bold', fontsize=11)
    axes[1, 1].axis('off')
    
    axes[1, 2].axis('off')
    info = f"""
    FFT ANALYSIS
    ════════════
    
    Image Size: {gray.shape}
    FFT Size: {fft.shape}
    
    Components:
    - DC Component (0,0):
      {magnitude_spectrum[magnitude_spectrum.shape[0]//2, 
                          magnitude_spectrum.shape[1]//2]:.2f}
    
    - Max Magnitude:
      {magnitude_spectrum.max():.2f}
    
    Properties:
    ✓ Center = Low freq
    ✓ Edges = High freq
    ✓ Brightness info
      at center
    ✓ Edge/detail info
      at periphery
    
    Uses:
    - Filtering
    - Compression
    - Analysis
    """
    axes[1, 2].text(0.1, 0.5, info, fontsize=9, family='monospace',
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.suptitle('9a. Fast Fourier Transform (FFT)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=True)
    
    print(f"Image size: {gray.shape}")
    print(f"FFT computed successfully")
    print(f"Magnitude range: [{magnitude_spectrum.min():.2f}, {magnitude_spectrum.max():.2f}]")
    print(f"DC component (mean): {magnitude_spectrum[magnitude_spectrum.shape[0]//2, magnitude_spectrum.shape[1]//2]:.2f}\n")
    
    return fft_shift, magnitude_spectrum, phase_spectrum


# ============================================================================
# Task 9b: Low Pass and High Pass Filtering in Frequency Domain
# ============================================================================
def frequency_domain_filtering(img, cutoff_low=30, cutoff_high=50):
    """
    Apply low pass and high pass filters in frequency domain
    """
    print("Task 9b: Low Pass and High Pass Filtering in Frequency Domain")
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    
    # Apply FFT
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    
    # Create filter masks
    # 1. Ideal Low Pass Filter
    lpf_mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(lpf_mask, (ccol, crow), cutoff_low, 1, -1)
    
    # 2. Ideal High Pass Filter
    hpf_mask = np.ones((rows, cols), np.uint8)
    cv2.circle(hpf_mask, (ccol, crow), cutoff_high, 0, -1)
    
    # 3. Butterworth Low Pass Filter
    def butterworth_lp(shape, cutoff, order=2):
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        u = np.arange(rows).reshape(-1, 1) - crow
        v = np.arange(cols) - ccol
        D = np.sqrt(u**2 + v**2)
        H = 1 / (1 + (D / cutoff) ** (2 * order))
        return H
    
    # 4. Butterworth High Pass Filter
    def butterworth_hp(shape, cutoff, order=2):
        return 1 - butterworth_lp(shape, cutoff, order)
    
    blpf_mask = butterworth_lp((rows, cols), cutoff_low, order=2)
    bhpf_mask = butterworth_hp((rows, cols), cutoff_high, order=2)
    
    # Apply filters
    fft_lpf_ideal = fft_shift * lpf_mask
    fft_hpf_ideal = fft_shift * hpf_mask
    fft_lpf_butter = fft_shift * blpf_mask
    fft_hpf_butter = fft_shift * bhpf_mask
    
    # Inverse FFT
    img_lpf_ideal = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_lpf_ideal)))
    img_hpf_ideal = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_hpf_ideal)))
    img_lpf_butter = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_lpf_butter)))
    img_hpf_butter = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_hpf_butter)))
    
    # Display
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Original and spectrum
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original Image', fontweight='bold', fontsize=10)
    axes[0, 0].axis('off')
    
    magnitude = np.log1p(np.abs(fft_shift))
    axes[0, 1].imshow(magnitude, cmap='hot')
    axes[0, 1].set_title('Magnitude Spectrum', fontweight='bold', fontsize=10)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(lpf_mask, cmap='gray')
    axes[0, 2].set_title(f'Ideal LPF Mask\n(Radius={cutoff_low})', fontweight='bold', fontsize=10)
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(hpf_mask, cmap='gray')
    axes[0, 3].set_title(f'Ideal HPF Mask\n(Radius={cutoff_high})', fontweight='bold', fontsize=10)
    axes[0, 3].axis('off')
    
    # Ideal filtered images
    axes[1, 0].imshow(img_lpf_ideal, cmap='gray')
    axes[1, 0].set_title('Ideal LPF Result\n(Smoothed)', fontweight='bold', fontsize=10)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(img_hpf_ideal, cmap='gray')
    axes[1, 1].set_title('Ideal HPF Result\n(Edges)', fontweight='bold', fontsize=10)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(blpf_mask, cmap='gray')
    axes[1, 2].set_title(f'Butterworth LPF\n(Cutoff={cutoff_low})', fontweight='bold', fontsize=10)
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(bhpf_mask, cmap='gray')
    axes[1, 3].set_title(f'Butterworth HPF\n(Cutoff={cutoff_high})', fontweight='bold', fontsize=10)
    axes[1, 3].axis('off')
    
    # Butterworth filtered images
    axes[2, 0].imshow(img_lpf_butter, cmap='gray')
    axes[2, 0].set_title('Butterworth LPF\n(Smooth Transition)', fontweight='bold', fontsize=10)
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(img_hpf_butter, cmap='gray')
    axes[2, 1].set_title('Butterworth HPF\n(Enhanced Edges)', fontweight='bold', fontsize=10)
    axes[2, 1].axis('off')
    
    # Filter profiles
    axes[2, 2].plot(lpf_mask[crow, :], 'b-', label='Ideal LPF', linewidth=2)
    axes[2, 2].plot(blpf_mask[crow, :], 'r-', label='Butterworth LPF', linewidth=2)
    axes[2, 2].set_title('LPF Profile', fontweight='bold', fontsize=10)
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)
    axes[2, 2].set_xlabel('Frequency')
    axes[2, 2].set_ylabel('Response')
    
    axes[2, 3].plot(hpf_mask[crow, :], 'b-', label='Ideal HPF', linewidth=2)
    axes[2, 3].plot(bhpf_mask[crow, :], 'r-', label='Butterworth HPF', linewidth=2)
    axes[2, 3].set_title('HPF Profile', fontweight='bold', fontsize=10)
    axes[2, 3].legend()
    axes[2, 3].grid(True, alpha=0.3)
    axes[2, 3].set_xlabel('Frequency')
    axes[2, 3].set_ylabel('Response')
    
    plt.suptitle('9b. Frequency Domain Filtering: Low Pass and High Pass', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=True)
    
    print("Frequency domain filtering completed:")
    print(f"Low pass cutoff: {cutoff_low}")
    print(f"High pass cutoff: {cutoff_high}")
    print("- Ideal filters: Sharp cutoff (ringing artifacts)")
    print("- Butterworth filters: Smooth transition (better results)\n")
    
    return img_lpf_butter, img_hpf_butter


# ============================================================================
# Task 9c: Apply IFFT to Reconstruct Image
# ============================================================================
def ifft_reconstruction(img):
    """
    Demonstrate FFT -> Modification -> IFFT reconstruction
    """
    print("Task 9c: Apply IFFT to Reconstruct Image")
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Forward FFT
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    
    # Scenario 1: Perfect reconstruction (no modification)
    reconstructed_perfect = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_shift)))
    
    # Scenario 2: Remove phase (keep only magnitude)
    magnitude = np.abs(fft_shift)
    reconstructed_magnitude_only = np.abs(np.fft.ifft2(np.fft.ifftshift(magnitude)))
    
    # Scenario 3: Remove magnitude (keep only phase)
    phase = np.exp(1j * np.angle(fft_shift))
    reconstructed_phase_only = np.abs(np.fft.ifft2(np.fft.ifftshift(phase)))
    
    # Scenario 4: Apply filter and reconstruct
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols))
    cv2.circle(mask, (ccol, crow), 60, 0, -1)  # Remove center (low freq)
    fft_filtered = fft_shift * mask
    reconstructed_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_filtered)))
    
    # Calculate reconstruction error
    error_perfect = np.abs(gray.astype(float) - reconstructed_perfect)
    
    # Display
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original Image', fontweight='bold', fontsize=10)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(np.log1p(np.abs(fft_shift)), cmap='hot')
    axes[0, 1].set_title('FFT Spectrum', fontweight='bold', fontsize=10)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(reconstructed_perfect, cmap='gray')
    axes[0, 2].set_title('Perfect Reconstruction\n(FFT→IFFT)', fontweight='bold', fontsize=10)
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(error_perfect, cmap='hot')
    axes[0, 3].set_title(f'Reconstruction Error\nMax={error_perfect.max():.2e}', 
                        fontweight='bold', fontsize=10)
    axes[0, 3].axis('off')
    
    axes[1, 0].imshow(reconstructed_magnitude_only, cmap='gray')
    axes[1, 0].set_title('Magnitude Only\n(Phase Removed)', fontweight='bold', fontsize=10)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(reconstructed_phase_only, cmap='gray')
    axes[1, 1].set_title('Phase Only\n(Magnitude Removed)', fontweight='bold', fontsize=10)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(reconstructed_filtered, cmap='gray')
    axes[1, 2].set_title('Low Freq Removed\n(High Pass Effect)', fontweight='bold', fontsize=10)
    axes[1, 2].axis('off')
    
    axes[1, 3].axis('off')
    info = """
    IFFT RECONSTRUCTION
    ══════════════════
    
    Process:
    Image → FFT → Modify → IFFT → Image
    
    Perfect Reconstruction:
    FFT → IFFT gives back
    original image
    
    Key Findings:
    
    ✓ Phase contains most
      structural info
    
    ✓ Magnitude contains
      contrast info
    
    ✓ Both needed for
      perfect reconstruction
    
    ✓ Small numerical errors
      in reconstruction
      (< 1e-10)
    
    Applications:
    - Filtering
    - Compression
    - Watermarking
    """
    axes[1, 3].text(0.05, 0.5, info, fontsize=8, family='monospace',
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    plt.suptitle('9c. Inverse FFT (IFFT) for Image Reconstruction', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=True)
    
    print("IFFT reconstruction analysis:")
    print(f"Perfect reconstruction error: {error_perfect.max():.2e}")
    print("Phase preserves structure better than magnitude")
    print("Both magnitude and phase needed for perfect reconstruction\n")
    
    return reconstructed_perfect


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================
def main_frequency_filtering():
    """Demonstrate frequency domain filtering"""
    print("="*70)
    print("PRACTICAL 9: FREQUENCY DOMAIN FILTERING")
    print("="*70 + "\n")
    
    # Create sample image
    print("Creating sample image...\n")
    img = np.zeros((400, 400), dtype=np.uint8)
    
    # Add features
    cv2.rectangle(img, (50, 50), (150, 150), 200, -1)
    cv2.circle(img, (300, 100), 50, 255, -1)
    cv2.rectangle(img, (100, 200), (350, 350), 150, -1)
    
    # Add fine details (high frequency)
    for i in range(0, 400, 10):
        cv2.line(img, (i, 0), (i, 400), 50, 1)
    
    cv2.imwrite('frequency_sample.jpg', img)
    
    print("Press Enter to start demonstrations...")
    input()
    
    # Execute all tasks
    # Task 9a: FFT
    fft_shift, magnitude, phase = apply_fft(img)
    
    # Task 9b: Frequency filtering
    lpf_result, hpf_result = frequency_domain_filtering(img, cutoff_low=30, cutoff_high=50)
    
    # Task 9c: IFFT reconstruction
    reconstructed = ifft_reconstruction(img)
    
    print("="*70)
    print("ALL FREQUENCY DOMAIN FILTERING TASKS COMPLETED!")
    print("="*70)


if __name__ == "__main__":
    main_frequency_filtering()