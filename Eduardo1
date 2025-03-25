# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 10:59:43 2025

@author: HUJ1GA
"""

import tensorflow as tf
import numpy as np
from pdf2image import convert_from_path
import cv2
from PIL import Image
import os

# Function to convert PDF to image
def pdf_to_image(pdf_path, output_path):
    pages = convert_from_path(pdf_path, dpi=200) # Adjust DPI for resolution
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        image_path = os.path.join(output_path, 'page.png')
        pages[0].save(image_path, 'PNG') # Save first page as PNG
        return image_path

# Function to preprocess image for TensorFlow
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
    img = cv2.resize(img, (512, 512)) # Resize for consistency
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32) / 255.0 # Normalize to [0, 1]
    return img_tensor

# Function to compare images and highlight differences
def compare_images(old_img_tensor, new_img_tensor, output_path):
    # Compute SSIM
    ssim = tf.image.ssim(old_img_tensor, new_img_tensor, max_val=1.0)  # SSIM values are between -1 and 1
    diff = 1.0 - ssim # Get the difference (0 means identical, 2 means completely different)

    # ... (rest of the compare_images function is the same)

def main():
    # ... (PDF paths, output directory)

    old_images = convert_from_path('00_Honda', dpi=200)
    new_images = convert_from_path('01_Honda', dpi=200)

    num_pages = min(len(old_images), len(new_images))

    for i in range(num_pages):
        old_img = np.array(old_images[i])
        new_img = np.array(new_images[i])

        if old_img.shape != new_img.shape:
            print(f"Page {i+1}: Dimensions differ, skipping comparison.")
            continue

        old_img_tensor = tf.convert_to_tensor(old_img, dtype=tf.float32) / 255.0
        new_img_tensor = tf.convert_to_tensor(new_img, dtype=tf.float32) / 255.0


        result_path = os.path.join(output_dir, f"differences_page_{i+1}.png")
        compare_images(old_img_tensor, new_img_tensor, result_path)


if __name__ == "__main__":
    main()
