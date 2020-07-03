# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 00:23:23 2019

@author: Monster
"""

# Importing necessary library 
import Augmentor 
import numpy as np
import cv2
import mask_generator as my
    
    
#Passing the paths of the image and labels directory 
d = my.mask_generator("C:/projects/python/tests/data_aug/train","C:/projects/python/tests/data_aug/train") 

# Passing the path of the image directory 
p = Augmentor.Pipeline("C:/projects/python/tests/data_aug/train", save_format="jpg") 
p.ground_truth("masks")

# Defining augmentation parameters and generating 500 samples 
p.flip_left_right(0.5)
p.flip_top_bottom(0.4) 
p.rotate(0.3, 10, 10) 
p.rotate_without_crop(0.3, 30, 30) 
p.random_brightness(0.5,0.6,1.25)
p.skew(0.4, 0.5) 
p.shear(0.3,10,10)
p.zoom(probability = 0.2, min_factor = 1.1, max_factor = 1.5) 
p.sample(500) 


d.rename('maskON')

#label name and label path
d.create_xml('maskON', 'train')

