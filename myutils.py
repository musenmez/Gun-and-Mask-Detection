import cv2
from pathlib import Path
from datetime import datetime



def detect_holding(_box, _fark,right_wrist, left_wrist):
    ymin = _box[0]
    xmin = _box[1]
    ymax = _box[2]
    xmax = _box[3]
    
    for m in range(len(right_wrist) if len(right_wrist) >= len(left_wrist) else len(left_wrist)):
        
        if right_wrist[m] != None:
            right_wrist_xmin = right_wrist[m][0] - _fark*2
            right_wrist_ymin = right_wrist[m][1] - _fark*2
            right_wrist_xmax = right_wrist[m][0] + _fark*2
            right_wrist_ymax = right_wrist[m][1] + _fark*2
            dx = min(right_wrist_xmax, xmax) - max(right_wrist_xmin, xmin)
            dy = min(right_wrist_ymax, ymax) - max(right_wrist_ymin, ymin)            
            if (dx>=0) and (dy>=0):
                return dx*dy, (right_wrist_ymin, right_wrist_xmin, right_wrist_ymax, right_wrist_xmax), m
            
        if left_wrist[m] != None:
            left_wrist_xmin = left_wrist[m][0] - _fark*2
            left_wrist_ymin = left_wrist[m][1] - _fark*2
            left_wrist_xmax = left_wrist[m][0] + _fark*2
            left_wrist_ymax = left_wrist[m][1] + _fark*2
            dx = min(left_wrist_xmax, xmax) - max(left_wrist_xmin, xmin)
            dy = min(left_wrist_ymax, ymax) - max(left_wrist_ymin, ymin)            
            if (dx>=0) and (dy>=0):
                return dx*dy, (left_wrist_ymin, left_wrist_xmin, left_wrist_ymax,  left_wrist_xmax),m
        else:
            return 0, (0,0,0,0), None

def detect_mask(_mask_box, face):
    ymin = _mask_box[0]
    xmin = _mask_box[1]
    ymax = _mask_box[2]
    xmax = _mask_box[3]
    
    for m in range(len(face)):
        if face[m] != None:
            start_point = face[m][0]
            end_point = face[m][1]       
            
            face_xmin = face[m][0][0]
            face_ymin = face[m][1][1]
            face_xmax = face[m][1][0]
            face_ymax = face[m][0][1]
            
            
            dx = min(face_xmax, xmax) - max(face_xmin, xmin)
            dy = min(face_ymax, ymax) - max(face_ymin, ymin)  
            if (dx>=0) and (dy>=0):
                face[m] = [start_point, end_point, 1]
            
    return face
        
   
        
def put_fps(_fps, _img):
    font = cv2.FONT_HERSHEY_SIMPLEX   
    # org 
    org = (50, 450) 
      
    # fontScale 
    fontScale = 1
       
    # Blue color in BGR 
    color = (255, 0, 0) 
      
    # Line thickness of 2 px 
    thickness = 2
       
    # Using cv2.putText() method 
    _img = cv2.putText(_img, 'FPS:' + _fps, org, font,  
                       fontScale, color, thickness, cv2.LINE_AA) 
    
    return _img

def create_main_folder(_archive):
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y__%H-%M-%S")
    Path(_archive + "/" + dt_string).mkdir(parents=True, exist_ok=True)
    return dt_string

def create_ID_folder(_archive, _folderName, _idNum):    
    Path(_archive + "/" + _folderName + "/" + "ID_" + _idNum).mkdir(parents=True, exist_ok=True)
    path = _archive + "/" + _folderName + "/" + "ID_" + _idNum
    return path
    
    
        