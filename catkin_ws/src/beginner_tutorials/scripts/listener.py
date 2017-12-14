#! /usr/bin/python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import os
from std_msgs.msg import String

# Instantiate CvBridge
bridge = CvBridge()

def lanesheep(frame):
    src = np.float32(
        [
            [581, 477],
            [699, 477],
            [896, 675],
            [384, 675]
        ]
    )
    # np.float32([[291, 234],[350, 234],[448, 338],[192,338]])
    # np.float32([256, 59, 144, 20])
    dst = np.float32(
        [
            [384, 0],
            [896, 0],
            [896, 720],
            [384, 720]
        ]
    )
    # np.float32([[192, 0],[448, 0],[448, 360],[192, 360]])
    # np.float32([0, 640, 360, 0])
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    frame = cv2.resize(frame, (1280, 720))
    # frame2 = cv2.warpPerspective(frame, M, (1280, 720), flags=cv2.INTER_LINEAR)
    # frame3 = cv2.warpPerspective(frame2, M_inv, (1280, 720), flags=cv2.INTER_LINEAR)

    hls = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)
    gray = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    sobel_binary = np.zeros(shape=gray.shape, dtype=bool)
    s_binary = sobel_binary
    combined_binary = s_binary.astype(np.float32)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=7)
    sobely = 0 #cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobel_abs = np.absolute(sobelx**2 + sobely**2)
    scaled_sobel = np.uint8(255*sobel_abs/np.max(sobel_abs))


    sobel_binary[(sobel_abs > 3) & (sobel_abs <= 255)] = 1

    s_binary[(s_channel >= 170) & (s_channel <= 255)] = 1

    combined_binary[(s_binary == 1) | (sobel_binary == 1)] = 1
    combined_binary = np.uint8(255 * combined_binary / np.max(combined_binary))
    # HSV [50,50,50] to HSV [110,255,255]
    lower_yellow = np.array([50,50,50])
    upper_yellow = np.array([110,255,255])

    lower_white = np.array([200,200,200])
    upper_white = np.array([255,255,255])
    mask = cv2.inRange(hls, lower_yellow, upper_yellow)
    mask2 = cv2.inRange(frame, lower_white, upper_white)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    res2 = cv2.bitwise_and(frame ,frame, mask= mask2)
    finres = cv2.bitwise_or(res ,res2)
    ret,finres = cv2.threshold(finres,1,255,cv2.THRESH_BINARY)
    frame2 = cv2.warpPerspective(finres, M, (1280, 720), flags=cv2.INTER_LINEAR)
    frame3 = cv2.warpPerspective(frame2, M_inv, (1280, 720), flags=cv2.INTER_LINEAR)
    # RGB [200,200,200] to RGB [255,255,255]

    # cv2.imshow("Stream: ", frame)
    # cv2.waitKey(0)
    # cv2.imshow("Transformed ", combined_binary)
    # cv2.waitKey(0)
    # cv2.imshow("Transformed ", finres)
    # cv2.waitKey(0)
    # cv2.imshow("Transformed ", frame2)
    # cv2.waitKey(0)
    blur = cv2.blur(frame3,(5,5))
    frame_cp = frame
    frame4 = cv2.Canny(blur, 50, 150, apertureSize=3)
    frame4 = cv2.warpPerspective(frame4, M, (1280, 720), flags=cv2.INTER_LINEAR)
    frame = cv2.warpPerspective(frame, M, (1280, 720), flags=cv2.INTER_LINEAR)

    lines = cv2.HoughLines(frame4,1,np.pi/180,150)
    x1_r = 0
    x2_r = 0
    y1_r = 0
    y2_r = 0
    n_r = 0
    x1_l = 0
    x2_l = 0
    y1_l = 0
    y2_l = 0
    n_l = 0
    if lines == None:
        return
    for line in lines :
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        # if x2==x1:
        #     # cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
        #     continue
        
        if x2<640 and x1<640:
            x1_l += x1
            x2_l += x2
            y1_l += y1
            y2_l += y2
            n_l += 1
            #cv2.line(frame,(x1, y1),(x2,y2),(0,0,255),2)
            continue
        
        if x2>640 and x1>640:
            x1_r += x1
            x2_r += x2
            y1_r += y1
            y2_r += y2
            n_r += 1
            continue
            cv2.line(frame,(x1, y1),(x2,y2),(0,0,255),2)
        
    if n_l==0:
        n_l = 1
    if n_r==0:
        n_r = 1
    cv2.line(frame,((x1_r/n_r),(y1_r)),(x2_r/n_r,y2_r),(0,255,0), 30)
    cv2.line(frame,((x1_l/n_l),(y1_l)),(x2_l/n_l,y2_l),(0,255,0),30)
    cv2.imshow("Output", cv2.warpPerspective(frame, M_inv, (1280, 720), flags=cv2.INTER_LINEAR))
    cv2.waitKey(1)


    pub = rospy.Publisher('custom_chatter', String)
    # rospy.init_node('custom_talker', anonymous=True)
    r = rospy.Rate(0.5) 


    while not rospy.is_shutdown():
        hello_str = "Slope 1: "+str((y1_r-y2_r)/(x1_r/n_r - x2_r/n_r))+"\nSlope 2:" + str((y1_l-y2_l)/(x1_l/n_l - x2_l/n_l))
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        r.sleep()

def image_callback(msg):
    print("Received an image!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError, e:
        print(e)
    else:
        # Save your OpenCV2 image as a jpeg 
        # cv2.imshow('camera_image.jpeg', cv2_img)
        # key = cv2.waitKey(1000)
        # if key == 27 or key == 1048603:
        #     cv2.destroyWindow("preview")
        lanesheep(cv2_img)

def main():
    rospy.init_node('image_listener')
    # Define your image topic
    image_topic = "video_frame"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, image_callback)
    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()