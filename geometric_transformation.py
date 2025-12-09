import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# PRACTICAL 6: GEOMETRIC TRANSFORMATIONS
# ============================================================================

# ============================================================================
# Task 6a: Translation
# ============================================================================
def translation(img, tx=50, ty=30):
    """
    Translate (shift) image by tx pixels horizontally and ty pixels vertically
    """
    print("Task 6a: Translation")
    
    # Convert to grayscale if needed for clarity
    if len(img.shape) == 3:
        display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        display_img = img
        
    rows, cols = img.shape[:2]
    
    # Create translation matrix
    # M = [[1, 0, tx],
    #      [0, 1, ty]]
    M_translation = np.float32([[1, 0, tx],
                                [0, 1, ty]])
    
    # Apply translation
    translated = cv2.warpAffine(img, M_translation, (cols, rows))
    
    if len(img.shape) == 3:
        translated_display = cv2.cvtColor(translated, cv2.COLOR_BGR2RGB)
    else:
        translated_display = translated
    
    # Display
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(display_img, cmap='gray' if len(img.shape)==2 else None)
    axes[0].set_title('Original Image', fontweight='bold', fontsize=12)
    axes[0].axis('off')
    axes[0].plot([cols//2], [rows//2], 'ro', markersize=10, label='Center')
    axes[0].legend()
    
    axes[1].imshow(translated_display, cmap='gray' if len(img.shape)==2 else None)
    axes[1].set_title(f'Translated (tx={tx}, ty={ty})', fontweight='bold', fontsize=12)
    axes[1].axis('off')
    axes[1].plot([cols//2 + tx], [rows//2 + ty], 'ro', markersize=10, label='New Center')
    axes[1].legend()
    
    plt.suptitle('6a. Translation Transformation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=True)
    
    print(f"Translation matrix:\n{M_translation}")
    print(f"Shifted by: ({tx}, {ty}) pixels\n")
    
    return translated, M_translation


# ============================================================================
# Task 6b: Scaling
# ============================================================================
def scaling(img, sx=1.5, sy=1.5):
    """
    Scale image by factors sx and sy
    """
    print("Task 6b: Scaling")
    
    if len(img.shape) == 3:
        display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        display_img = img
        
    rows, cols = img.shape[:2]
    
    # Method 1: Using cv2.resize
    new_size = (int(cols * sx), int(rows * sy))
    scaled_resize = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
    
    # Method 2: Using transformation matrix
    M_scale = np.float32([[sx, 0, 0],
                          [0, sy, 0]])
    scaled_warp = cv2.warpAffine(img, M_scale, new_size)
    
    if len(img.shape) == 3:
        scaled_display = cv2.cvtColor(scaled_resize, cv2.COLOR_BGR2RGB)
    else:
        scaled_display = scaled_resize
    
    # Display
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(display_img, cmap='gray' if len(img.shape)==2 else None)
    axes[0].set_title(f'Original ({cols}x{rows})', fontweight='bold', fontsize=11)
    axes[0].axis('off')
    
    axes[1].imshow(scaled_display, cmap='gray' if len(img.shape)==2 else None)
    axes[1].set_title(f'Scaled (sx={sx}, sy={sy})\n{new_size[0]}x{new_size[1]}', 
                     fontweight='bold', fontsize=11)
    axes[1].axis('off')
    
    # Show both scaling up and down
    scale_down = cv2.resize(img, (cols//2, rows//2), interpolation=cv2.INTER_AREA)
    if len(img.shape) == 3:
        scale_down_display = cv2.cvtColor(scale_down, cv2.COLOR_BGR2RGB)
    else:
        scale_down_display = scale_down
    
    axes[2].imshow(scale_down_display, cmap='gray' if len(img.shape)==2 else None)
    axes[2].set_title(f'Scaled Down (0.5x)\n{cols//2}x{rows//2}', 
                     fontweight='bold', fontsize=11)
    axes[2].axis('off')
    
    plt.suptitle('6b. Scaling Transformation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=True)
    
    print(f"Scaling matrix:\n{M_scale}")
    print(f"Original size: {cols}x{rows}")
    print(f"Scaled size: {new_size[0]}x{new_size[1]}\n")
    
    return scaled_resize, M_scale


# ============================================================================
# Task 6c: Rotation
# ============================================================================
def rotation(img, angle=45):
    """
    Rotate image by specified angle (degrees)
    """
    print("Task 6c: Rotation")
    
    if len(img.shape) == 3:
        display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        display_img = img
        
    rows, cols = img.shape[:2]
    
    # Calculate center point
    center = (cols // 2, rows // 2)
    
    # Get rotation matrix
    M_rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation
    rotated = cv2.warpAffine(img, M_rotation, (cols, rows))
    
    # Rotation with different angles
    rotated_90 = cv2.warpAffine(img, cv2.getRotationMatrix2D(center, 90, 1.0), (cols, rows))
    rotated_180 = cv2.warpAffine(img, cv2.getRotationMatrix2D(center, 180, 1.0), (cols, rows))
    
    if len(img.shape) == 3:
        rotated_display = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
        rotated_90_display = cv2.cvtColor(rotated_90, cv2.COLOR_BGR2RGB)
        rotated_180_display = cv2.cvtColor(rotated_180, cv2.COLOR_BGR2RGB)
    else:
        rotated_display = rotated
        rotated_90_display = rotated_90
        rotated_180_display = rotated_180
    
    # Display
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(display_img, cmap='gray' if len(img.shape)==2 else None)
    axes[0, 0].set_title('Original', fontweight='bold', fontsize=11)
    axes[0, 0].plot([center[0]], [center[1]], 'ro', markersize=8)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(rotated_display, cmap='gray' if len(img.shape)==2 else None)
    axes[0, 1].set_title(f'Rotated {angle}°', fontweight='bold', fontsize=11)
    axes[0, 1].plot([center[0]], [center[1]], 'ro', markersize=8)
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(rotated_90_display, cmap='gray' if len(img.shape)==2 else None)
    axes[1, 0].set_title('Rotated 90°', fontweight='bold', fontsize=11)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(rotated_180_display, cmap='gray' if len(img.shape)==2 else None)
    axes[1, 1].set_title('Rotated 180°', fontweight='bold', fontsize=11)
    axes[1, 1].axis('off')
    
    plt.suptitle('6c. Rotation Transformation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=True)
    
    print(f"Rotation matrix:\n{M_rotation}")
    print(f"Center of rotation: {center}")
    print(f"Rotation angle: {angle}°\n")
    
    return rotated, M_rotation


# ============================================================================
# Task 6d: Shrinking
# ============================================================================
def shrinking(img, shrink_factor=0.5):
    """
    Shrink image by reducing its size
    """
    print("Task 6d: Shrinking")
    
    if len(img.shape) == 3:
        display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        display_img = img
        
    rows, cols = img.shape[:2]
    
    # Method 1: Using resize with INTER_AREA (best for shrinking)
    new_size = (int(cols * shrink_factor), int(rows * shrink_factor))
    shrunken_area = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    
    # Method 2: Using different interpolations
    shrunken_linear = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
    shrunken_cubic = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)
    
    # Different shrink factors
    shrink_25 = cv2.resize(img, (cols//4, rows//4), interpolation=cv2.INTER_AREA)
    
    if len(img.shape) == 3:
        shrunken_display = cv2.cvtColor(shrunken_area, cv2.COLOR_BGR2RGB)
        shrink_25_display = cv2.cvtColor(shrink_25, cv2.COLOR_BGR2RGB)
    else:
        shrunken_display = shrunken_area
        shrink_25_display = shrink_25
    
    # Display
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(display_img, cmap='gray' if len(img.shape)==2 else None)
    axes[0, 0].set_title(f'Original\n{cols}x{rows}', fontweight='bold', fontsize=11)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(shrunken_display, cmap='gray' if len(img.shape)==2 else None)
    axes[0, 1].set_title(f'Shrunken (factor={shrink_factor})\n{new_size[0]}x{new_size[1]}', 
                        fontweight='bold', fontsize=11)
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(shrink_25_display, cmap='gray' if len(img.shape)==2 else None)
    axes[1, 0].set_title(f'Shrunken (factor=0.25)\n{cols//4}x{rows//4}', 
                        fontweight='bold', fontsize=11)
    axes[1, 0].axis('off')
    
    # Show comparison of interpolation methods
    axes[1, 1].axis('off')
    info_text = f"""
    SHRINKING METHODS
    ═══════════════════
    
    Original Size: {cols}x{rows}
    Shrunken Size: {new_size[0]}x{new_size[1]}
    
    Shrink Factor: {shrink_factor}
    Reduction: {(1-shrink_factor)*100:.0f}%
    
    Best Interpolation for Shrinking:
    - INTER_AREA (default)
    
    Other Methods:
    - INTER_LINEAR
    - INTER_CUBIC
    - INTER_NEAREST
    
    Use Case:
    Creating thumbnails,
    reducing file size,
    downsampling
    """
    axes[1, 1].text(0.1, 0.5, info_text, fontsize=10, family='monospace',
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.suptitle('6d. Shrinking Transformation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=True)
    
    print(f"Original size: {cols}x{rows}")
    print(f"Shrunken size: {new_size[0]}x{new_size[1]}")
    print(f"Shrink factor: {shrink_factor}")
    print(f"Size reduction: {(1-shrink_factor)*100:.0f}%\n")
    
    return shrunken_area


# ============================================================================
# Task 6e: Zooming
# ============================================================================
def zooming(img, zoom_factor=2.0, zoom_region=None):
    """
    Zoom into image by enlarging it or a specific region
    """
    print("Task 6e: Zooming")
    
    if len(img.shape) == 3:
        display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        display_img = img
        
    rows, cols = img.shape[:2]
    
    # Method 1: Zoom entire image
    new_size = (int(cols * zoom_factor), int(rows * zoom_factor))
    zoomed_full = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)
    
    # Method 2: Zoom into specific region (region of interest)
    if zoom_region is None:
        # Default: zoom into center quarter
        start_row, end_row = rows//4, 3*rows//4
        start_col, end_col = cols//4, 3*cols//4
    else:
        start_row, end_row, start_col, end_col = zoom_region
    
    roi = img[start_row:end_row, start_col:end_col]
    zoomed_roi = cv2.resize(roi, (cols, rows), interpolation=cv2.INTER_CUBIC)
    
    # Create image with ROI marked
    img_with_roi = img.copy()
    cv2.rectangle(img_with_roi, (start_col, start_row), (end_col, end_row), 
                  (255, 0, 0) if len(img.shape)==3 else 255, 3)
    
    if len(img.shape) == 3:
        zoomed_display = cv2.cvtColor(zoomed_full, cv2.COLOR_BGR2RGB)
        zoomed_roi_display = cv2.cvtColor(zoomed_roi, cv2.COLOR_BGR2RGB)
        roi_marked = cv2.cvtColor(img_with_roi, cv2.COLOR_BGR2RGB)
    else:
        zoomed_display = zoomed_full
        zoomed_roi_display = zoomed_roi
        roi_marked = img_with_roi
    
    # Display
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(display_img, cmap='gray' if len(img.shape)==2 else None)
    axes[0, 0].set_title(f'Original\n{cols}x{rows}', fontweight='bold', fontsize=11)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(zoomed_display, cmap='gray' if len(img.shape)==2 else None)
    axes[0, 1].set_title(f'Zoomed Full (factor={zoom_factor})\n{new_size[0]}x{new_size[1]}', 
                        fontweight='bold', fontsize=11)
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(roi_marked, cmap='gray' if len(img.shape)==2 else None)
    axes[1, 0].set_title('ROI Marked (Red Rectangle)', fontweight='bold', fontsize=11)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(zoomed_roi_display, cmap='gray' if len(img.shape)==2 else None)
    axes[1, 1].set_title('Zoomed ROI\n(Enlarged to original size)', 
                        fontweight='bold', fontsize=11)
    axes[1, 1].axis('off')
    
    plt.suptitle('6e. Zooming Transformation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=True)
    
    print(f"Original size: {cols}x{rows}")
    print(f"Zoomed full size: {new_size[0]}x{new_size[1]}")
    print(f"Zoom factor: {zoom_factor}")
    print(f"ROI region: [{start_row}:{end_row}, {start_col}:{end_col}]\n")
    
    return zoomed_full, zoomed_roi


# ============================================================================
# ALL TRANSFORMATIONS IN ONE VIEW
# ============================================================================
def all_transformations_combined(img):
    """Display all geometric transformations together"""
    print("\nCombined View: All Geometric Transformations")
    
    if len(img.shape) == 3:
        display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        display_img = img
        
    rows, cols = img.shape[:2]
    center = (cols // 2, rows // 2)
    
    # Apply all transformations
    # Translation
    M_trans = np.float32([[1, 0, 50], [0, 1, 30]])
    translated = cv2.warpAffine(img, M_trans, (cols, rows))
    
    # Scaling
    scaled = cv2.resize(img, None, fx=0.7, fy=0.7)
    
    # Rotation
    M_rot = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated = cv2.warpAffine(img, M_rot, (cols, rows))
    
    # Shrinking
    shrunken = cv2.resize(img, (cols//2, rows//2), interpolation=cv2.INTER_AREA)
    
    # Zooming
    zoomed = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    
    # Convert for display
    if len(img.shape) == 3:
        trans_disp = cv2.cvtColor(translated, cv2.COLOR_BGR2RGB)
        rot_disp = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
    else:
        trans_disp = translated
        rot_disp = rotated
    
    # Display
    fig = plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(display_img, cmap='gray' if len(img.shape)==2 else None)
    plt.title('Original', fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(trans_disp, cmap='gray' if len(img.shape)==2 else None)
    plt.title('Translation (50, 30)', fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(scaled, cmap='gray' if len(img.shape)==2 else None)
    plt.title('Scaling (0.7x)', fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(rot_disp, cmap='gray' if len(img.shape)==2 else None)
    plt.title('Rotation (45°)', fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(shrunken, cmap='gray' if len(img.shape)==2 else None)
    plt.title('Shrinking (0.5x)', fontweight='bold')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(zoomed, cmap='gray' if len(img.shape)==2 else None)
    plt.title('Zooming (1.5x)', fontweight='bold')
    plt.axis('off')
    
    plt.suptitle('All Geometric Transformations Combined', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show(block=True)


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================
def main_geometric_transformations():
    """Demonstrate all geometric transformations"""
    print("="*70)
    print("PRACTICAL 6: GEOMETRIC TRANSFORMATIONS")
    print("="*70 + "\n")
    
    # Create sample image
    print("Creating sample image...\n")
    img = np.zeros((400, 400), dtype=np.uint8)
    
    # Add patterns
    cv2.rectangle(img, (50, 50), (150, 150), 200, -1)
    cv2.circle(img, (300, 100), 50, 255, -1)
    cv2.rectangle(img, (200, 200), (350, 350), 150, -1)
    cv2.putText(img, 'TRANSFORM', (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    
    cv2.imwrite('geometric_sample.jpg', img)
    
    print("Press Enter to start demonstrations...")
    input()
    
    # Execute all tasks
    translated, M_trans = translation(img, tx=50, ty=30)
    scaled, M_scale = scaling(img, sx=1.5, sy=1.5)
    rotated, M_rot = rotation(img, angle=45)
    shrunken = shrinking(img, shrink_factor=0.5)
    zoomed_full, zoomed_roi = zooming(img, zoom_factor=2.0)
    
    # Show all transformations together
    all_transformations_combined(img)
    
    print("="*70)
    print("ALL GEOMETRIC TRANSFORMATIONS COMPLETED!")
    print("="*70)
    print("\nTransformation Matrices:")
    print(f"Translation:\n{M_trans}\n")
    print(f"Scaling:\n{M_scale}\n")
    print(f"Rotation:\n{M_rot}\n")


if __name__ == "__main__":
    main_geometric_transformations()