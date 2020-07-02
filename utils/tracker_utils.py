# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 23:50:41 2020

@author: Monster
"""



# USAGE
# python opencv_object_tracking.py
# python opencv_object_tracking.py --video dashcam_boston.mp4 --tracker csrt

# import the necessary packages

from imutils.video import FPS
import argparse
import imutils
import cv2


def tracker_on(frame, initBB = None, con = False):
# construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str,
    	help="path to input video file")
    ap.add_argument("-t", "--tracker", type=str, default="csrt",
    	help="OpenCV object tracker type")
    args = vars(ap.parse_args())
    
    # extract the OpenCV version info
    (major, minor) = cv2.__version__.split(".")[:2]
    
    # if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
    # function to create our object tracker
    if int(major) == 3 and int(minor) < 3:
    	tracker = cv2.Tracker_create(args["tracker"].upper())
    
    # otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
    # approrpiate object tracker constructor:
    else:
    	# initialize a dictionary that maps strings to their corresponding
    	# OpenCV object tracker implementations
    	OPENCV_OBJECT_TRACKERS = {
    		"csrt": cv2.TrackerCSRT_create,
    		"kcf": cv2.TrackerKCF_create,
    		"boosting": cv2.TrackerBoosting_create,
    		"mil": cv2.TrackerMIL_create,
    		"tld": cv2.TrackerTLD_create,
    		"medianflow": cv2.TrackerMedianFlow_create,
    		"mosse": cv2.TrackerMOSSE_create
    	}
    
    	# grab the appropriate object tracker using our dictionary of
    	# OpenCV object tracker objects
    	tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
    
    # initialize the bounding box coordinates of the object we are going
    # to track
     
    
    # initialize the FPS throughput estimator
    fps = None

	# check to see if we have reached the end of the stream
    if frame is None:
        return
    
    if con == False:
    	# resize the frame (so we can process it faster) and grab the
    	# frame dimensions
        frame = imutils.resize(frame, width=500)
        (H, W) = frame.shape[:2]
        
        # select the bounding box of the object we want to track (make
    	# sure you press ENTER or SPACE after selecting the ROI)
        
        
        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        tracker.init(frame, initBB)
        fps = FPS().start()


	# check to see if we are currently tracking an object
    if con == True:
		# grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)

		# check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),
				(0, 255, 0), 2)

		# update the FPS counter
        fps.update()
        fps.stop()

		# initialize the set of information we'll be displaying on
		# the frame
        info = [
			("Tracker", args["tracker"]),
			("Success", "Yes" if success else "No"),
			("FPS", "{:.2f}".format(fps.fps())),
		]

		# loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
    return frame
	
   
	



 
        