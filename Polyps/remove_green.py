import cv2
import numpy as np
import os
import glob

def auto_detect_and_replace_green(image_path, output_path):
    """Automatically detect green rectangle and replace with black."""
    img = cv2.imread(image_path)
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Use BGR instead of RGB
    
    # Define green color range
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # Create mask of green areas
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Find contours of green areas
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Fill detected green areas with black
    for contour in contours:
        if cv2.contourArea(contour) > 5000:
            x, y, w, h = cv2.boundingRect(contour)
            img[y:y+h, x:x+w] = [0, 0, 0]
    
    # Save modified image
    filename = os.path.basename(image_path)
    save_path = os.path.join(output_path, f"{filename}")
    cv2.imwrite(save_path, img)

# Process all images in the folder
input_folder = '/home/luca/Desktop/images'
output_folder = '/home/luca/Desktop/images/'
os.makedirs(output_folder, exist_ok=True)

for image_file in glob.glob(os.path.join(input_folder, '*.jpg')):  # Adjust extension if needed
    auto_detect_and_replace_green(image_file, output_folder)
