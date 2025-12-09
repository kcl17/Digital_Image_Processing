import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================================
# Task 1a: Read and Display Image
# ============================================================================
def read_and_display_image(image_path):
    """Read and display an image"""
    print("Task 1a: Reading and Displaying Image")
    
    # Read image
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return None
    
    # Convert BGR to RGB for display (OpenCV reads in BGR format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Display
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.title('Original Image - Close this window to continue')
    plt.axis('off')
    plt.show(block=True)
    
    print(f"Image shape: {img.shape}")
    print(f"Image dtype: {img.dtype}\n")
    
    return img


# ============================================================================
# Task 1b: Resize Image
# ============================================================================
def resize_image(img, width=300, height=300):
    """Resize image to specified dimensions"""
    print("Task 1b: Resizing Image")
    
    # Resize image
    resized = cv2.resize(img, (width, height))
    
    # Display comparison
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title(f'Original ({img.shape[1]}x{img.shape[0]})')
    axes[0].axis('off')
    
    axes[1].imshow(resized_rgb)
    axes[1].set_title(f'Resized ({width}x{height})')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show(block=True)
    
    print(f"Original size: {img.shape}")
    print(f"Resized to: {resized.shape}\n")
    
    return resized


# ============================================================================
# Task 1c: Convert Color to Grayscale
# ============================================================================
def color_to_grayscale(img):
    """Convert color image to grayscale"""
    print("Task 1c: Converting to Grayscale")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Display comparison
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title('Color Image')
    axes[0].axis('off')
    
    axes[1].imshow(gray, cmap='gray')
    axes[1].set_title('Grayscale Image')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show(block=True)
    
    print(f"Color image shape: {img.shape}")
    print(f"Grayscale image shape: {gray.shape}\n")
    
    return gray


# ============================================================================
# Task 1d: Convert to Black & White (Binary)
# ============================================================================
def convert_to_bw(img, threshold=127):
    """Convert image to black and white (binary)"""
    print("Task 1d: Converting to Black & White")
    
    # Convert to grayscale first if it's a color image
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply binary threshold
    _, bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Display
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(gray, cmap='gray')
    axes[0].set_title('Grayscale Image')
    axes[0].axis('off')
    
    axes[1].imshow(bw, cmap='gray')
    axes[1].set_title(f'Black & White (threshold={threshold})')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show(block=True)
    
    print(f"Unique values in B&W image: {np.unique(bw)}\n")
    
    return bw


# ============================================================================
# Task 1e: Separate RGB Planes
# ============================================================================
def separate_rgb_planes(img):
    """Separate color image into R, G, B planes"""
    print("Task 1e: Separating RGB Planes")
    
    # Split channels (OpenCV uses BGR format)
    b, g, r = cv2.split(img)
    
    # Display all planes
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original Color Image')
    axes[0, 0].axis('off')
    
    # Red plane
    axes[0, 1].imshow(r, cmap='Reds')
    axes[0, 1].set_title('Red Plane')
    axes[0, 1].axis('off')
    
    # Green plane
    axes[1, 0].imshow(g, cmap='Greens')
    axes[1, 0].set_title('Green Plane')
    axes[1, 0].axis('off')
    
    # Blue plane
    axes[1, 1].imshow(b, cmap='Blues')
    axes[1, 1].set_title('Blue Plane')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Red plane shape: {r.shape}")
    print(f"Green plane shape: {g.shape}")
    print(f"Blue plane shape: {b.shape}\n")
    
    return r, g, b


# ============================================================================
# Task 1f: Create Color Image from RGB Planes
# ============================================================================
def create_color_from_planes(r, g, b):
    """Create color image from separate R, G, B planes"""
    print("Task 1f: Creating Color Image from RGB Planes")
    
    # Merge channels (OpenCV uses BGR format)
    img_bgr = cv2.merge([b, g, r])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Display
    plt.figure(figsize=(8, 6))
    plt.imshow(img_rgb)
    plt.title('Color Image Created from R, G, B Planes')
    plt.axis('off')
    plt.show()
    
    print(f"Created color image shape: {img_bgr.shape}\n")
    
    return img_bgr


# ============================================================================
# Task 1g: Write 2D Data to Image File
# ============================================================================
def write_2d_data_to_image(data, filename='output_image.png'):
    """Write 2D numpy array data to image file"""
    print("Task 1g: Writing 2D Data to Image File")
    
    # Normalize data to 0-255 range if needed
    if data.dtype != np.uint8:
        data_normalized = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
    else:
        data_normalized = data
    
    # Write image
    cv2.imwrite(filename, data_normalized)
    
    # Verify by reading back
    read_back = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    
    # Display
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(data_normalized, cmap='gray')
    axes[0].set_title('Original 2D Data')
    axes[0].axis('off')
    
    axes[1].imshow(read_back, cmap='gray')
    axes[1].set_title(f'Image Read from File: {filename}')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"2D data shape: {data.shape}")
    print(f"Image saved as: {filename}")
    print(f"File written successfully!\n")


# ============================================================================
# Task 1f (Profile): Draw Image Profile
# ============================================================================
def draw_image_profile(img, row_idx=None, col_idx=None):
    """Draw intensity profile along a row or column"""
    print("Task 1f (Profile): Drawing Image Profile")
    
    # Convert to grayscale if color
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Default to middle row and column
    if row_idx is None:
        row_idx = gray.shape[0] // 2
    if col_idx is None:
        col_idx = gray.shape[1] // 2
    
    # Get profiles
    row_profile = gray[row_idx, :]
    col_profile = gray[:, col_idx]
    
    # Display
    fig = plt.figure(figsize=(14, 10))
    
    # Original image with lines
    ax1 = plt.subplot(2, 2, 1)
    plt.imshow(gray, cmap='gray')
    plt.axhline(y=row_idx, color='r', linestyle='--', label=f'Row {row_idx}')
    plt.axvline(x=col_idx, color='b', linestyle='--', label=f'Col {col_idx}')
    plt.title('Image with Profile Lines')
    plt.legend()
    plt.axis('off')
    
    # Row profile
    ax2 = plt.subplot(2, 2, 2)
    plt.plot(row_profile, 'r-', linewidth=2)
    plt.title(f'Horizontal Profile (Row {row_idx})')
    plt.xlabel('Column Index')
    plt.ylabel('Intensity')
    plt.grid(True, alpha=0.3)
    
    # Column profile
    ax3 = plt.subplot(2, 2, 3)
    plt.plot(col_profile, 'b-', linewidth=2)
    plt.title(f'Vertical Profile (Column {col_idx})')
    plt.xlabel('Row Index')
    plt.ylabel('Intensity')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Row profile shape: {row_profile.shape}")
    print(f"Column profile shape: {col_profile.shape}\n")


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================
def main():
    """Main function to demonstrate all tasks"""
    print("="*70)
    print("IMAGE PROCESSING PRACTICAL - ALL TASKS")
    print("="*70 + "\n")
    
    # Create a sample image for demonstration
    print("Creating sample image for demonstration...\n")
    
    # Create a colorful sample image
    sample_img = np.zeros((300, 300, 3), dtype=np.uint8)
    sample_img[:100, :, 2] = 255  # Red top
    sample_img[100:200, :, 1] = 255  # Green middle
    sample_img[200:, :, 0] = 255  # Blue bottom
    
    # Add some patterns
    cv2.circle(sample_img, (150, 150), 50, (255, 255, 0), -1)
    cv2.rectangle(sample_img, (50, 50), (250, 250), (255, 255, 255), 3)
    
    cv2.imwrite('sample_image.jpg', sample_img)
    
    # Execute all tasks
    # Task 1a
    img = read_and_display_image('sample_image.jpg')
    
    if img is not None:
        # Task 1b
        resized = resize_image(img, 200, 200)
        
        # Task 1c
        gray = color_to_grayscale(img)
        
        # Task 1d
        bw = convert_to_bw(img, threshold=127)
        
        # Task 1f (Profile)
        draw_image_profile(img)
        
        # Task 1e
        r, g, b = separate_rgb_planes(img)
        
        # Task 1f
        reconstructed = create_color_from_planes(r, g, b)
        
        # Task 1g
        # Create sample 2D data (gradient)
        sample_data = np.linspace(0, 255, 100*100).reshape(100, 100)
        write_2d_data_to_image(sample_data, 'output_2d_data.png')
        
        print("="*70)
        print("ALL TASKS COMPLETED SUCCESSFULLY!")
        print("="*70)

if __name__ == "__main__":
    main()