# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 02:15:17 2019

@author: Monster
"""



import numpy as np
import cv2
import xml.etree.ElementTree as ET
import os

#os.getcwd()

class mask_generator():
    def __init__(self, image_directory, xml_directory):
        self.image_directory = image_directory
        self.xml_directory = xml_directory
        self.current_path = os.getcwd()
        
        if not os.path.exists('masks') :
            os.mkdir('masks')
        if not os.path.exists('output_xml') :
            os.mkdir('output_xml')
            
        for xml_file in os.listdir(self.xml_directory): 
            
            if os.path.isdir(os.path.join(self.xml_directory, xml_file)):                
                continue
            
            file_type = xml_file.split('.', 1)[1]
            file_rest = xml_file.split('.', 1)[0]
            
            if file_type == 'xml':            
                image_file = file_rest + '.jpg'
                img = cv2.imread(self.image_directory + '/' + image_file)
                h = np.size(img,0)
                w = np.size(img,1)              
                tree = ET.parse(self.xml_directory + '/' + xml_file)
                root = tree.getroot()
                mask = np.zeros((h, w))
                
                for box in root.iter('bndbox'):
                    xmin = int(box.find('xmin').text)
                    ymin = int(box.find('ymin').text)        
                    xmax = int(box.find('xmax').text)
                    ymax = int(box.find('ymax').text)                      
                    mask[ymin:ymax, xmin:xmax] = 255
                    
                cv2.imwrite('masks/' + image_file, mask)
            else:
                continue
    
    def rename(self, save_name):        
        self.save_name = save_name
        collection = self.image_directory + '/output'         
        if not os.path.exists(os.path.join(self.current_path, 'output_images')):
            os.mkdir('output_images')
        if not os.path.exists(os.path.join(self.current_path, 'output_masks')):
            os.mkdir('output_masks')
        for i, filename in enumerate(os.listdir(collection)):             
            file_head = filename.split('_', 4)[1]
            file_tail = filename.split('.', 1)[1]              
            if file_head == 'original':
                os.rename(collection + '/' + filename, self.current_path + "/output_images/" + self.save_name + str(i) + ".jpg")        
            
            for m, subname in enumerate(os.listdir(collection)):                 
                sub_head = subname.split('_',5)[1]
                sub_tail = subname.split('.', 1)[1]
                if sub_head == 'groundtruth' and sub_tail == file_tail:
                    os.rename(collection + '/' + subname, self.current_path + "/output_masks/" + self.save_name + str(i) + ".jpg")

    
    def create_xml(self, object_name, folder_name):
        self.object_name = object_name    
        self.folder_name = folder_name
        base_xml = "base.xml"
        i=0
        collection = "output_masks"
        for fileName in os.listdir(collection):     
            print(i)
            i=i+1
            filename_rest = fileName.split('.', 1)[0]    
            mask = cv2.imread(collection +'/' + fileName)  
            gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            ret,thresh = cv2.threshold(gray_mask,90,255,0) 
            contours, _  = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)   
            ht = np.size(thresh,0)
            wt = np.size(thresh,1) 
            if not contours:
                os.remove(collection +'/' + fileName)
                os.remove("output_images/"+ filename_rest + ".jpg")
                continue
            
            cnt = contours[0]
            area = cv2.contourArea(cnt)           
            
            x,y,w,h = cv2.boundingRect(cnt) 
            xmin = x
            ymin = y
            xmax = x+w
            ymax = y+h
            
            if (ht==ymax and area < 20000) or (0==ymin and area < 20000) or (wt==xmax and area < 20000) or (0==xmin and area < 20000) or (area < 500):
                os.remove(collection +'/' + fileName)
                os.remove("output_images/"+ filename_rest + ".jpg")
                continue
            
            #xml düzenleme
            
            
            filename_rest = fileName.split('.', 1)[0]            
            tree = ET.parse(base_xml)
            root = tree.getroot()
            
            new_folder = folder_name
            path_root = 'C:\\tensorflow1\\models\\research\\object_detection\\images\\' + new_folder
            
            new_name = filename_rest+'.xml'
            new_path = path_root + '\\' + filename_rest + '.jpg'
            
            
            for filename in root.iter('name'): 
                filename.text = object_name
            
            for filename in root.iter('filename'):    
                filename.text = filename_rest + '.jpg'
            for path in root.iter('path'):    
                path.text = new_path  
            
            for folder in root.iter('folder'):
                folder.text = new_folder
                
            for box in root.iter('bndbox'):        
                box.find('xmin').text = str(xmin)
                box.find('ymin').text = str(ymin)   
                box.find('xmax').text = str(xmax)
                box.find('ymax').text = str(ymax)
            tree.write('output_xml/'+ new_name)
           

 

        




