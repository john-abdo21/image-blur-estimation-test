# -*- coding: utf-8 -*-
"""
Created on Fri May 26 13:51:08 2023

@author: Administrator
"""

import cv2
import numpy as np


def GetImageWrittenText(img, text, font = cv2.FONT_HERSHEY_SIMPLEX, font_scale = 1, thickness = 2, text_margin = 5, back_color = (255, 255, 255) , text_color = (0, 0, 255)):
    
    # Get the dimensions of the image
    height, width = img.shape[:2]
    
    # Define the text string and font properties
    
    # Get the size of the text
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Get the width and height of the text
    text_width, text_height = text_size
    
    # Set the desired height of the colored space
    colored_space_height = text_height + text_margin * 2
    
    # Create a new image with the desired height and width
    new_img = np.zeros((height + colored_space_height, width, 3), dtype=np.uint8)
    
    # Fill the new space with the desired color
    
    new_img[height:, :] = back_color
    
    # Copy the original image onto the new image
    new_img[:height, :] = img
    
    
    # Define the position of the text
    text_pos = (int((width - text_width)/2), height + text_height + text_margin)
    
    # Draw the text on the image
    cv2.putText(new_img, text, text_pos, font, font_scale, text_color, thickness, cv2.LINE_AA)
    
    # return new image
    return new_img


# Load the image
img = cv2.imread('image.jpg')

# Get text written image

new_img = GetImageWrittenText(img, "this is my text", font = cv2.FONT_HERSHEY_PLAIN, font_scale = 1, thickness = 1)


# Display the resulting image
cv2.imshow('New Image', new_img)
cv2.waitKey(0)
