#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pydicom
from pydicom import dcmread
from pydicom.uid import ExplicitVRLittleEndian

import pandas as pd
from pandas import DataFrame

import numpy as np



from numba import jit
import cv2

from skimage.transform import rescale

import os


# In[2]:


def import_dicom(in_dir_path):
    #names the path to the input model directory, and lists the files in this directory

    in_dir_list = sorted(os.listdir(in_dir_path))
    model_depth = len(in_dir_list)
    model_data = []


    i = 0
    while i < model_depth:
        path = in_dir_path + "/" + in_dir_list[i]
        current_slice = pydicom.dcmread(path)
        model_slice = current_slice.pixel_array
        model_slice = np.asarray(model_slice)
        model_data.append(model_slice)
        i = i + 1
    model_data = np.asarray(model_data, dtype = np.int16)
    
        
    return model_data


# In[3]:


@jit
def add_img(img_1, img_2):
    
    #tests which of the inputs is the larger image
    if len(img_1) > len(img_2):
        lar_img = img_1
        sml_img = img_2
    else:
        lar_img = img_2
        sml_img = img_1

    #finds the distance (in the x-axis direction) between the left
    #corners of the two images, if the smaller image was placed
    #in the middle of the larger image
    lar_centre_index_x = round(len(lar_img)/2) - 1
    sml_centre_index_x = round(len(sml_img)/2) - 1
    edge_x = lar_centre_index_x - sml_centre_index_x
       
    #finds the distance (in the y-axis direction) between the left
    #corners of the two images, if the smaller image was placed
    #in the middle of the larger image
    lar_centre_index_y = round(len(lar_img[0])/2) - 1
    sml_centre_index_y = round(len(sml_img[0])/2) - 1
    edge_y = lar_centre_index_y - sml_centre_index_y

    output = lar_img
    
    x = edge_x
    
    #works through each pixel where the two images overlap
    #and adds the value of the pixels from each image together
    while x < (edge_x + len(sml_img)):
        i = edge_y     
        while i < (edge_y + len(sml_img[0])):
                out_pixel = (lar_img[x][i] + sml_img[x - edge_x][i - edge_y])
                output[x][i]= out_pixel            
                i = i + 1
        x = x + 1
    
    return output
    


# In[4]:


@jit
def project(model, source_object_distance, object_image_distance):
    source_image_distance = source_object_distance + len(model) + object_image_distance
    
    model_depth = len(model)
    model_height = len(model[0])
    model_length = len(model[0][0])
    
    
    max_mag = ((source_image_distance)/(source_object_distance))

    

    base_cube = np.zeros((model_depth, int(np.ceil(max_mag * model_height)), int(np.ceil(max_mag * model_length))))
    enclosed_cube = np.zeros((model_depth, int(np.ceil(max_mag * model_height)), int(np.ceil(max_mag * model_length))))
  
    i = 0
    while i < len(model):
        slice_mag = (source_image_distance)/(source_object_distance + i)
        temp = cv2.resize(model[i], dsize = (round(model_length * slice_mag), round(model_height * slice_mag)), interpolation = cv2.INTER_LINEAR)
        temp = temp/(slice_mag*slice_mag)
        
        enclosed_cube[i] = add_img(base_cube[i], temp)
        
        i = i + 1
        
    
    projection = np.sum(base_cube, axis = 0)
    return projection
    


# In[5]:


def save_dicom(image, filename):
    dicom_out_path = get_testdata_file("CT_small.dcm")
    dicom_out = dcmread(path)
    dicom_out.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dicom_out.PixelData = image.astype(np.float16).tobytes()
    dicom_out.save_as("filename.dcm")
    


# In[6]:


import matplotlib.pyplot as plt


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




