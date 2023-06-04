# -*- coding: utf-8 -*-
"""
Created on Sat May 27 07:25:15 2023

@author: Administrator
"""

import cv2
import os
import csv
import numpy as np
from skimage import filters
image_extensions=['.jpg','.bmp','.jpeg']

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

def blur_estimate(basename, extension):

    
    filename = basename + extension    
    print (filename)
    
    # Load an image and estimate the blur
    image = cv2.imread(filename)
    
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert the image to grayscale
    denoised_gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    
    adjusted_gray = clahe.apply(gray)
    
    adjusted_denoised_gray = clahe.apply(denoised_gray)

    # # Create a CLAHE object with default parameters
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    # # Apply CLAHE to the grayscale image
    # clahe_gray = clahe.apply(gray)
    
    # # Apply histogram equalization to the grayscale image
    # equalized_clahe_gray = cv2.equalizeHist(clahe_gray)
    
    # # Apply Gaussian blur to remove some noise
    
    # blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # clahe_blurred = cv2.GaussianBlur(clahe_gray, (5, 5), 0)
    # equalized_clahe_blurred = cv2.GaussianBlur(equalized_clahe_gray, (5, 5), 0)
    
    # # Compute the Fourier transform of the image and shift the zero-frequency component to the center
    # f = np.fft.fft2(gray)
    # fshift = np.fft.fftshift(f)

    # # Apply a high-pass filter to the Fourier transform to remove low-frequency components
    # rows, cols = gray.shape
    # crow, ccol = int(rows/2), int(cols/2)
    # fshift[crow-30:crow+30, ccol-30:ccol+30] = 0

    # # Shift the zero-frequency component back to the top-left corner
    # f_ishift = np.fft.ifftshift(fshift)

    # # Compute the inverse Fourier transform to obtain the filtered image
    # filtered = np.fft.ifft2(f_ishift).real

    # # Compute the Laplacian of the filtered image to estimate the degree of blur
    # laplacian = cv2.Laplacian(filtered, cv2.CV_64F)
    # variance = np.var(laplacian)
    # blur_metric = variance
    
    # Compute the edges using the Canny edge detection algorithm
    edges = filters.sobel(denoised_gray)
    
    # Compute the mean edge intensity
    mean_edge_intensity = np.mean(edges) # edges.mean()
    
    # Compute the blur metric from the mean edge intensity
    edge_blur_metric = 1 / mean_edge_intensity
    
    
    # Define the threshold values
    threshold1 = 100
    threshold2 = 200
    
    # Apply Canny Edge Detection
    edges = cv2.Canny(denoised_gray, threshold1, threshold2)
    
    # Calculate the edge intensity
    edge_intensity = cv2.sumElems(edges)
    
    
    
    
    # Compute the Laplacian variance
    
    lap_var = {}
    lap_var['original'] = cv2.Laplacian(image, cv2.CV_64F).var()
    lap_var['denoised'] = cv2.Laplacian(denoised_image, cv2.CV_64F).var()
    lap_var['gray'] = cv2.Laplacian(gray, cv2.CV_64F).var()
    lap_var['denoisedgray'] = cv2.Laplacian(denoised_gray, cv2.CV_64F).var()
    lap_var['adjustedgray'] = cv2.Laplacian(adjusted_gray, cv2.CV_64F).var()
    lap_var['adjusteddenoisedgray'] = cv2.Laplacian(adjusted_denoised_gray, cv2.CV_64F).var()
    lap_var['edgeblurmetric']=edge_blur_metric
    lap_var['edgeintensity']=edge_intensity[0]
    # lap_var['blurred'] = cv2.Laplacian(blurred, cv2.CV_64F).var()
    # lap_var['gray_blurred'] = cv2.Laplacian(gray_blurred, cv2.CV_64F).var()
    # lap_var['clahe_grey'] = cv2.Laplacian(clahe_gray, cv2.CV_64F).var()
    # lap_var['equalized_clahe_gray'] = cv2.Laplacian(equalized_clahe_gray, cv2.CV_64F).var()
    # lap_var['clahe_blurred'] = cv2.Laplacian(clahe_blurred, cv2.CV_64F).var()
    # lap_var['equalized_clahe_blurred'] = cv2.Laplacian(equalized_clahe_blurred, cv2.CV_64F).var()
    # lap_var['fft']= blur_metric
    
    #output = cv2.hconcat([gray, clahe_gray, equalized_clahe_gray])
    
    denoised_gray=GetImageWrittenText(denoised_gray, "denoised gray image", font = cv2.FONT_HERSHEY_PLAIN, font_scale = 1, thickness = 1)
    adjusted_gray=GetImageWrittenText(adjusted_gray, "adjusted gray image", font = cv2.FONT_HERSHEY_PLAIN, font_scale = 1, thickness = 1)
    output = cv2.hconcat([denoised_gray, adjusted_gray])
    
    cv2.imwrite(f"{working_dir}/{basename}{extension}", output)
    if lap_var['adjustedgray']>200 and lap_var['edgeintensity']>3000000:
        new_imsge = GetImageWrittenText(image, "this is Correct image", font = cv2.FONT_HERSHEY_PLAIN, font_scale = 1, thickness = 1)
        cv2.imwrite(f"{correct_dir}/{basename}{extension}", new_imsge)
    else:
        new_imsge = GetImageWrittenText(image, "this is blurred image", font = cv2.FONT_HERSHEY_PLAIN, font_scale = 1, thickness = 1)
        cv2.imwrite(f"{blurred_dir}/{basename}{extension}", new_imsge)
    # cv2.imwrite(f"{working_dir}/{basename}_autolevel{extension}", equalized)   
    # cv2.imwrite(f"{working_dir}/{basename}_autocontrast{extension}", clahe_image)   
    # if (lap_var['original']>100):
    #     cv2.imwrite(f"{working_dir}/{basename}{extension}", image)   
    #     cv2.imwrite(f"{working_dir}/{basename}_gray{extension}", gray)   
    #     cv2.imwrite(f"{working_dir}/{basename}_blurred{extension}", blurred) 
    #     cv2.imwrite(f"{working_dir}/{basename}_grey_blurred{extension}", gray_blurred) 
    # else:
    #     cv2.imwrite(f"{blurred_dir}/{basename}{extension}", image)   
    #     cv2.imwrite(f"{blurred_dir}/{basename}_gray{extension}", gray)   
    #     cv2.imwrite(f"{blurred_dir}/{basename}_blurred{extension}", blurred) 
    #     cv2.imwrite(f"{blurred_dir}/{basename}_grey_blurred{extension}", gray_blurred)

    # Return the blur estimation
    return lap_var

  

#os.chdir('../testimage')
#os.chdir('../sampleimage/real_20230522')
#prefix="test_"
prefix="real_"
# prefix=""
os.chdir(f"../sampleimage/{prefix}20230522")

working_dir = "processedImage"
correct_dir = "CorrectImage"
blurred_dir = "blurredImage"

if not os.path.exists(working_dir):
    os.mkdir(working_dir)
if not os.path.exists(blurred_dir):
    os.mkdir(blurred_dir)
if not os.path.exists(correct_dir):
    os.mkdir(correct_dir)
img_filenames=os.listdir()
csv_filename = f"{working_dir}/laplacian_{prefix}test-6.csv"
data=[]
data.append(["filename","original","denoised","gray","denoisedgray","adjustedgray","adjusteddenoisedgray","edgeblurmetric","edgeintensity"])
#data.append(["filename","blurred"])

i=0
for filename in img_filenames:
    basename, extension = os.path.splitext(filename)
    if extension in image_extensions:
        i = i+1
        # if i>10:
        #     break
        
        blur = blur_estimate(basename, extension)
        
        d=[filename]
        d.extend(list(blur.values()))
        
        data.append (d)
        
# writing to CSV file
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)
