#!/usr/bin/env python


import rospy
import cv2
import sys
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError 
import time


def image_publisher():
    # Publishes to the topic video_frame
    pub = rospy.Publisher('video_frame', Image, queue_size=100)
    rospy.init_node('image_publisher', anonymous=True)
    # Refreshes once in 2 seconds
    rate = rospy.Rate(0.5)
    # Bridge module is for coverting ROS image type to opencv and vice versa
    bridge = CvBridge()
    resource = '/home/advait/catkin_ws/src/beginner_tutorials/scripts/lane_detection.py/lanevid.mov'
    cap = cv2.VideoCapture(resource)
    if not cap.isOpened():
        print ("Error opening resource: " + str(resource))
        exit(0)


    print ("Correctly opened resource, starting to show feed.")
    rval, frame = cap.read()
    while rval:
        cv2.imshow("Stream: " + resource, frame)
        rval, frame = cap.read()

        # If frame exists, use ROS function to convert to ros message
        if frame is not None:
            frame = np.uint8(frame)
        image_message = bridge.cv2_to_imgmsg(frame, "bgr8")

        pub.publish(image_message)
        print('Published succesfully!')


        key = cv2.waitKey(1000)
        if key == 27 or key == 1048603:
            break
    cv2.destroyWindow("preview")




if __name__ == '__main__':
    try:
        image_publisher()
    except rospy.ROSInterruptException:
        pass
