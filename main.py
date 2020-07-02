
# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
import argparse
import posenet
from datetime import datetime
from pyimagesearch.centroidtracker import CentroidTracker
import telegram #pytthon-telegram-bot
import requests
from PyQt5.QtCore import QThread


sys.path.append("..")

# Import utilites

from utils import label_map_util
from utils import visualization_utils as vis_util
import myutils

#check net connection
def check_internet():
    url='http://www.google.com/'
    timeout=5
    try:
        _ = requests.get(url, timeout=timeout)
        return True
    except requests.ConnectionError:
        print("No Internet Connection.")
    return False


class ThreadClass(QThread):
    def __init__(self, chatid, token, _imageName, parent = None):
        super(ThreadClass, self).__init__(parent)  
        self.imageName = _imageName
        self.chatid = chatid
        self.token = token
        self.bot = telegram.Bot(token=self.token)        
    
    def run(self):      
        print("run")    
        if check_internet():
            self.bot.send_message(self.chatid, "Şüpheli Şahıs")
            self.bot.send_photo(chat_id=self.chatid, photo=open(self.imageName, 'rb'), timeout=100)  
      
              

def main(cameraID, chatID, token, model):
    print("Start Point")
    main.faceList_ = None
    main.isStopped = 0       
    main.e = 0
    
    e = check_camera(cameraID)
    if(e == 404):
        main.e = e
        return e
    
    e = validate_token(token)
    if(e == 101):
        main.e = e
        return e    
    
    start = time.time() 
    tlg = ThreadClass(chatID, token, "placeHolder")  
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=int, default=101)
    parser.add_argument('--cam_id', type=int, default=0)
    parser.add_argument('--cam_width', type=int, default=500)
    parser.add_argument('--cam_height', type=int, default=500)
    parser.add_argument('--scale_factor', type=float, default=1) #0.7125
    parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
    args = parser.parse_args()
    
    # initialize our centroid tracker and frame dimensions
    ct = CentroidTracker()
    treatID=[None]*100
    IDvPose=[None]*100
    objectID = None
    
    
    #Myutils Things
    archive = "archive"
    folderName = myutils.create_main_folder(archive)
    
    
    if model == 1:        
        # Name of the directory containing the object detection module we're using
        MODEL_NAME = 'inference_graph_gun'
        
        
        # Grab path to current working directory
        CWD_PATH = os.getcwd()
        
        # Path to frozen detection graph .pb file, which contains the model that is used
        # for object detection.
        PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
        
        
        # Path to label map file
        PATH_TO_LABELS = os.path.join(CWD_PATH,'inference_graph_gun','labelmap.pbtxt')
        
        
        # Number of classes the object detector can identify
        NUM_CLASSES = 1
        
    elif model == 2:
        # Name of the directory containing the object detection module we're using
        MODEL_NAME = 'inference_graph_mask'
        
        
        # Grab path to current working directory
        CWD_PATH = os.getcwd()
        
        # Path to frozen detection graph .pb file, which contains the model that is used
        # for object detection.
        PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
        
        
        # Path to label map file
        PATH_TO_LABELS = os.path.join(CWD_PATH,'inference_graph_mask','labelmap.pbtxt')
        
        
        # Number of classes the object detector can identify
        NUM_CLASSES = 2
    
    elif model == 3:
        # Name of the directory containing the object detection module we're using
        MODEL_NAME = 'inference_graph_maskgun'
        
        
        # Grab path to current working directory
        CWD_PATH = os.getcwd()
        
        # Path to frozen detection graph .pb file, which contains the model that is used
        # for object detection.
        PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
        
        
        # Path to label map file
        PATH_TO_LABELS = os.path.join(CWD_PATH,'inference_graph_maskgun','labelmap.pbtxt')
        
        
        # Number of classes the object detector can identify
        NUM_CLASSES = 3
        
    
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
    
    
    
    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()    
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
        
    
    
    
    
    # Define input and output tensors (i.e. data) for the object detection classifier
    
    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    
    
    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    
    
    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    
    
    
    
    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    
    with tf.Session() as sess2:
        model_cfg, model_outputs = posenet.load_model(args.model, sess2)
        output_stride = model_cfg['output_stride']
        
        if args.file is not None:
            cap = cv2.VideoCapture(args.file)
        else:        
            cap = cv2.VideoCapture(cameraID)#args.cam_id
        cap.set(3, args.cam_width) #640-480
        cap.set(4, args.cam_height)
        height,width = 480,640
    
        while(True):
            fps = time.time()           
            
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride) 
        
        
        
            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess2.run(
                    model_outputs,
                    feed_dict={'image:0': input_image}
                )           
            
            
            
            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)
            
            keypoint_coords *= output_scale
            overlay_image = display_image        
            frame_expanded = np.expand_dims(overlay_image, axis=0)
            # Input tensor is the image
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            
            
            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded})
            
            
            
            
            # Draw the results of the detection (aka 'visulaize the results')
            vis_util.visualize_boxes_and_labels_on_image_array(
                overlay_image,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.85)
            
            
            b= []
            rects = []
            
            overlay_image, b = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords, b,
                min_pose_score=0.15, min_part_score=0.1)
            
            IDvPose=[None]*100
            faceList=[None]*100
            mask_status=[None]*100
            nose = [None]*10
            left_eye = [None]*10
            right_eye = [None]*10
            left_wrist = [None]*10
            right_wrist = [None]*10
            face = [None]*10
            fark = 0
            
            
            for i in range(len(b)):
                if b[i] == []:
                    continue
                for m in range(len(b[i])):
                    
                    if b[i][m][0] == 0:
                        nose[i] = int(b[i][m][1][1]),int(b[i][m][1][0])
                        
                    if b[i][m][0] == 2:
                        left_eye[i] = int(b[i][m][1][1]),int(b[i][m][1][0])
                    
                    if b[i][m][0] == 1:
                        right_eye[i] = int(b[i][m][1][1]),int(b[i][m][1][0])
                        
                    if b[i][m][0] == 9:
                        left_wrist[i] = int(b[i][m][1][1]),int(b[i][m][1][0])
                    
                    if b[i][m][0] == 10:
                        right_wrist[i] = int(b[i][m][1][1]),int(b[i][m][1][0])
                    
            
            for i in range(len(nose)):   
                if nose[i] != None and left_eye[i] !=  None  and right_eye[i] !=  None:
                    fark = abs(left_eye[i][0]-right_eye[i][0])
                    start_pointX= nose[i][1] + int(fark*1.5)
                    start_pointY =int(left_eye[i][0] - fark)
                    end_pointX = nose[i][1] - int(fark*2.5)
                    end_pointY = int(right_eye[i][0] + fark)
                    
                    if start_pointX > 480:
                        start_pointX = 480
                    if start_pointY < 0:
                        start_pointY = 0
                    if end_pointX < 0:
                        end_pointX = 0
                    if end_pointY > 640:
                        end_pointY = 640
                    
                    start_point = (start_pointY, start_pointX)
                    end_point= (end_pointY, end_pointX)   
                        
                    overlay_image = cv2.rectangle(overlay_image, start_point, end_point, color=(0,150,0), thickness=2)  
                    face[i] = [start_point, end_point, 0]      
                    startY = min(face[i][0][0],face[i][1][0])
                    endY = max(face[i][0][0],face[i][1][0])
                    startX = min(face[i][0][1],face[i][1][1])
                    endX = max(face[i][0][1],face[i][1][1])
                    box = np.array([startY, startX, endY, endX, i]) #Y ile X ters why?
                    rects.append(box.astype("int"))
            
            objects = ct.update(rects)
            
            
            
            
            
            
            if model == 1:
                a = ()
                #Koordinat Çekme    
                max_boxes_to_draw=20    
                if not max_boxes_to_draw:
                    max_boxes_to_draw = np.squeeze(boxes).shape[0]
                for i in range(min(max_boxes_to_draw, np.squeeze(boxes).shape[0])):
                    if np.squeeze(scores) is None or np.squeeze(scores)[i] > 0.8: 
                        box = tuple(np.squeeze(boxes)[i].tolist())
                        ymin, xmin, ymax, xmax = box
                        box = (int(ymin*height)), (int(xmin*width)), (int(ymax*height)), (int(xmax*width))                        
                        isHolding, a, num = myutils.detect_holding(box, fark, right_wrist, left_wrist)                        
                        if isHolding != 0:  
                            if num != None:
                                if face[num] == None:
                                    continue                      
                                for (objectID, centroid) in objects.items():
                                    if num == centroid[2]:
                                        treatID[objectID] = objectID 
            
            elif model == 2:
                a = ()
                #Koordinat Çekme    
                max_boxes_to_draw=20    
                if not max_boxes_to_draw:
                    max_boxes_to_draw = np.squeeze(boxes).shape[0]
                for i in range(min(max_boxes_to_draw, np.squeeze(boxes).shape[0])):
                    if np.squeeze(scores) is None or np.squeeze(scores)[i] > 0.8:
                        class_name = np.squeeze(classes)[i].tolist() #maskON = 1.0
                        if class_name == 1.0:                    
                            mask_box = tuple(np.squeeze(boxes)[i].tolist())
                            ymin, xmin, ymax, xmax = mask_box
                            mask_box = (int(ymin*height)), (int(xmin*width)), (int(ymax*height)), (int(xmax*width))   
                            face = myutils.detect_mask(mask_box, face)                             
                        
            
            elif model == 3:
                a = ()
                #Koordinat Çekme    
                max_boxes_to_draw=20    
                if not max_boxes_to_draw:
                    max_boxes_to_draw = np.squeeze(boxes).shape[0]
                for i in range(min(max_boxes_to_draw, np.squeeze(boxes).shape[0])):
                    if np.squeeze(scores) is None or np.squeeze(scores)[i] > 0.8:
                        class_name = np.squeeze(classes)[i].tolist() #maskON = 2.0
                        if class_name == 2.0:                    
                            mask_box = tuple(np.squeeze(boxes)[i].tolist())
                            ymin, xmin, ymax, xmax = mask_box
                            mask_box = (int(ymin*height)), (int(xmin*width)), (int(ymax*height)), (int(xmax*width))   
                            face = myutils.detect_mask(mask_box, face) 
                            
                        if class_name == 1.0:    #gun                
                            box = tuple(np.squeeze(boxes)[i].tolist())
                            ymin, xmin, ymax, xmax = box
                            box = (int(ymin*height)), (int(xmin*width)), (int(ymax*height)), (int(xmax*width))                        
                            isHolding, a, num = myutils.detect_holding(box, fark, right_wrist, left_wrist)
                            
                            if isHolding != 0:  
                                if num != None:
                                    if face[num] == None:
                                        continue                      
                                    for (objectID, centroid) in objects.items():
                                        if num == centroid[2]:
                                            treatID[objectID] = objectID
            
            
            # loop over the tracked objects
            for (objectID, centroid) in objects.items():
            		# draw both the ID of the object and the centroid of the
            		# object on the output frame                
                IDvPose[objectID] = centroid[2]   
                faceList[objectID] = face[centroid[2]]
                text = "ID {}".format(objectID)
                cv2.putText(overlay_image, text, (centroid[0] - 10, centroid[1] - 10),
            			 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                      
            for i in range(len(treatID)):
                if treatID[i] == None:
                    continue    
                
                num = IDvPose[treatID[i]]  
                
                if num != None:  
                    # font 
                    font = cv2.FONT_HERSHEY_SIMPLEX                   
                    # org 
                    org = (50, 50)                   
                    # fontScale 
                    fontScale = 1                   
                    # Blue color in BGR 
                    color = (0, 0, 255)                   
                    # Line thickness of 2 px 
                    thickness = 2                    
                    overlay_image = cv2.putText(overlay_image, 'Threat detected', org, font,  
                                                fontScale, color, thickness, cv2.LINE_AA)                 
                    if face[num] == None:
                        continue  
                    
                    #Create ID folder
                    pathName = myutils.create_ID_folder(archive, folderName, str(treatID[i]))                
                    
                    start_point = face[num][0]
                    end_point = face[num][1]
                    overlay_image = cv2.rectangle(overlay_image, start_point, end_point, color=(0,0,255), thickness=2)  
                    ymin = min(face[num][0][0],face[num][1][0])
                    ymax = max(face[num][0][0],face[num][1][0])
                    xmin = min(face[num][0][1],face[num][1][1])
                    xmax = max(face[num][0][1],face[num][1][1])
                    input_image, display_image, output_scale = posenet.read_cap(
                    cap, scale_factor=args.scale_factor, output_stride=output_stride) 
                    crop_img = display_image[xmin:xmax,ymin:ymax]
                    now = datetime.now()
                    dt_string = now.strftime("%d-%m-%Y__%H-%M-%S")
                    img_name = pathName + "\\Suspect_" + dt_string + ".jpg"                
                    cv2.imwrite(img_name, crop_img) 
                    if time.time() - start > 10:
                       start = time.time()
                       tlg.imageName = img_name
                       tlg.start()
                
                              
                 
                  
            pfps = time.time() - fps
            pfps = str(int(1/pfps))
            myutils.put_fps(pfps, overlay_image)    
            input_image, display_image, output_scale = posenet.read_cap(
                    cap, scale_factor=args.scale_factor, output_stride=output_stride) 
            faceList, mask_status = crop_face(IDvPose, face, display_image,faceList, mask_status)
            
            main.overlay_image_ = overlay_image
            main.cap_ = cap
            main.faceList_ = faceList   
            main.mask_status_ = mask_status            
            
            if main.isStopped == 1:
                print("Break Point")
                break
            
            
            
            
def crop_face(IDvPose, face, display_image,faceList, mask_status):
    for i in range(len(IDvPose)):                 
                
        num = IDvPose[i]          
        if num != None:                           
            if face[num] == None:
                continue   
            else:             
                ymin = min(face[num][0][0],face[num][1][0])
                ymax = max(face[num][0][0],face[num][1][0])
                xmin = min(face[num][0][1],face[num][1][1])
                xmax = max(face[num][0][1],face[num][1][1])                    
                crop_img2 = display_image[xmin:xmax,ymin:ymax]   
                faceList[i] = crop_img2     
                mask_status[i] = face[num][2] 
                if int(crop_img2.shape[0]) <= 20 and int(crop_img2.shape[1]) <=20:
                    faceList[i] = None     
                    mask_status[i] = None               
    return faceList, mask_status
                    
            

def get_all_values():
    if hasattr(main, 'overlay_image_') and hasattr(main, 'cap_'):
        return main.overlay_image_, main.cap_, main.faceList_, main.mask_status_
    else:    
        main.overlay_image_ = cv2.imread("gui_images/cctvholder.png")
        main.cap_ = None
        main.mask_status_ = None
        return main.overlay_image_, main.cap_, main.faceList_, main.mask_status_
    
def check_stop(x):    
    main.isStopped = x
    print("check_stop: ", main.isStopped)
    return main.isStopped

def check_camera(x):
    cap = cv2.VideoCapture(x)
    if cap is None or not cap.isOpened():
        print("hata")
        return 404
    else:
        return 1
    
def validate_token(token):
        """A very basic validation on token."""
        if any(x.isspace() for x in token):
            return 101

        left, sep, _right = token.partition(':')
        if (not sep) or (not left.isdigit()) or (len(left) < 3):
            return 101
        
        return 0
    
def get_error():
    return main.e
 
    

    
    
        


        
        
                 
           
            
