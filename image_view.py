from __future__ import print_function
from __future__ import division

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os


# BLACK & WHITE IMAGING

# img = cv.imread('ramen_packet.png',cv2.IMREAD_GRAYSCALE)
# cv.imshow('image',img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# BACKGROUND SUBTRACTION METHOD

# parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
#                                               OpenCV. You can process both videos and images.')
# parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='IMG_4271.MOV')
# parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
# args = parser.parse_args()
# if args.algo == 'MOG2':
#     backSub = cv.createBackgroundSubtractorMOG2()
# else:
#     backSub = cv.createBackgroundSubtractorKNN()
# capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
# if not capture.isOpened():
#     print('Unable to open: ' + args.input)
#     exit(0)
# while True:
#     ret, frame = capture.read()
#     if frame is None:
#         break
    
#     fgMask = backSub.apply(frame)
    
    
#     cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
#     cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
#                cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    
#     cv.imshow('Frame', frame)
#     cv.imshow('FG Mask', fgMask)
    
#     keyboard = cv.waitKey(30)
    
# MEANSHIFT AND CAMSHIFT
    
# parser = argparse.ArgumentParser(description='This sample demonstrates the meanshift algorithm. \
#                                               The example file can be downloaded from: \
#                                               https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
# parser.add_argument('--input', type=str, help='path to image file',default='IMG_4271.MOV') 
# args = parser.parse_args()
# cap = cv.VideoCapture(args.input)

# # take first frame of the video
# ret,frame = cap.read()

# # setup initial location of window
# x, y, w, h = 300, 200, 100, 50 # simply hardcoded the values
# track_window = (x, y, w, h)

# # set up the ROI for tracking
# roi = frame[y:y+h, x:x+w]
# hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
# mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
# roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
# cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

# # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
# term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
# while(1):
#     ret, frame = cap.read()
#     if ret == True:
#         hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#         dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        
#         # apply meanshift to get the new location
#         ret, track_window = cv.meanShift(dst, track_window, term_crit)
        
#         # Draw it on image
# #         x,y,w,h = track_window
# #         img2 = cv.rectangle(frame, (x,y), (x+w,y+h), 255,2)
# #         cv.imshow('img2',img2)
# #         k = cv.waitKey(30) & 0xff
# #         if k == 27:
# #             break
# #     else:
# #         break


# # OPTICAL FLOW


# parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
#                                               The example file can be downloaded from: \
#                                               https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
# parser.add_argument('--input', type=str, help='path to image file',default='IMG_4271.MOV')
# args = parser.parse_args()
# cap = cv.VideoCapture(args.input)
# # params for ShiTomasi corner detection
# feature_params = dict( maxCorners = 100,
#                        qualityLevel = 0.3,
#                        minDistance = 7,
#                        blockSize = 7 )
# # Parameters for lucas kanade optical flow
# lk_params = dict( winSize  = (15,15),
#                   maxLevel = 2,
#                   criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# # Create some random colors
# color = np.random.randint(0,255,(100,3))
# # Take first frame and find corners in it
# ret, old_frame = cap.read()
# old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
# p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# # Create a mask image for drawing purposes
# mask = np.zeros_like(old_frame)
# while(1):
#     ret,frame = cap.read()
#     frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     # calculate optical flow
#     p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
#     # Select good points
#     if p1 is not None:
#         good_new = p1[st==1]
#         good_old = p0[st==1]
#     # draw the tracks
#     for i,(new,old) in enumerate(zip(good_new, good_old)):
#         a,b = new.ravel()
#         c,d = old.ravel()
#         mask = cv.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
#         frame = cv.circle(frame,(int(a),int(b)),5,color[i].tolist(),-1)
#     img = cv.add(frame,mask)
#     cv.imshow('frame',img)
#     k = cv.waitKey(30) & 0xff
#     if k == 27:
#         break
#     # Now update the previous frame and previous points
#     old_gray = frame_gray.copy()
#     p0 = good_new.reshape(-1,1,2)
    
    
# DENSE OPTICAL FLOW

cap = cv.VideoCapture(cv.samples.findFile("IMG_4271.MOV"))
ret, frame1 = cap.read()
prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255
while(1):
    ret, frame2 = cap.read()
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    cv.imshow('frame2',bgr)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv.imwrite('opticalfb.png',frame2)
        cv.imwrite('opticalhsv.png',bgr)
    prvs = next