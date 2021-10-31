#!/usr/bin/env python3

from __future__ import division

# Python imports
import numpy as np
import scipy.io as sio
import os, sys, cv2, time
from torch.cuda import is_available

# ROS imports
import rospy
import std_msgs.msg
from rospkg import RosPack
from std_msgs.msg import UInt8
from sensor_msgs.msg import Image
from geometry_msgs.msg import Polygon, Point32
#from cv_bridge import CvBridge, CvBridgeError

package = RosPack()
package_path = package.get_path('yolov3_pytorch_ros')

# Deep learning imports
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models.models import Darknet
from utils.utils import *
from vision_msgs.msg import ObjectHypothesisWithPose, Detection2D, Detection2DArray

# Detector manager class for YOLO
class DetectorManager():
    def __init__(self):
        # Load weights parameter
        weights_name = rospy.get_param('~weights_name', 'yolov3.weights')
        self.weights_path = os.path.join(package_path, 'models', weights_name)
        rospy.loginfo("Found weights, loading %s", self.weights_path)

        # Raise error if it cannot find the model
        if not os.path.isfile(self.weights_path):
            raise IOError(('{:s} not found.').format(self.weights_path))

        # Load image parameter and confidence threshold
        self.image_topic = rospy.get_param('~image_topic', '/camera/rgb/image_raw')
        self.confidence_th = rospy.get_param('~confidence', 0.7)
        self.nms_th = rospy.get_param('~nms_th', 0.3)

        # Load publisher topics
        self.detected_objects_topic = rospy.get_param('~detected_objects_topic')
        self.published_image_topic = rospy.get_param('~detections_image_topic')

        # Load other parameters
        config_name = rospy.get_param('~config_name', 'yolov3.cfg')
        self.config_path = os.path.join(package_path, 'config', config_name)
        classes_name = rospy.get_param('~classes_name', 'coco.names')
        self.classes_path = os.path.join(package_path, 'classes', classes_name)
        self.gpu_id = rospy.get_param('~gpu_id', 0)
        self.network_img_size = rospy.get_param('~img_size', 416)
        self.publish_image = rospy.get_param('~publish_image')
        self.use_cuda = rospy.get_param('~use_cuda', True)
        
        # Initialize width and height
        self.h = 0
        self.w = 0

        rospy.loginfo("config path: " + self.config_path)
        self.model = Darknet(self.config_path)
        # Load net
        if self.weights_path.endswith('.pth'):
            self.model.load_state_dict(torch.load(self.weights_path))
        else:
            self.model.load_darknet_weights(self.weights_path)

        if torch.cuda.is_available() and self.use_cuda:
            rospy.loginfo("CUDA available, use GPU")
            self.device = torch.device('cuda')
            self.model.cuda()
        else:
            self.device = torch.device('cpu')
            rospy.loginfo("CUDA not available, use CPU")
            # if CUDA not available, use CPU
            # self.checkpoint = torch.load(self.weights_path, map_location=torch.device('cpu'))
            # self.model.load_state_dict(self.checkpoint)
        self.model.eval() # Set in evaluation mode
        rospy.loginfo("Deep neural network loaded")

        # Load CvBridge
        #self.bridge = CvBridge()

        # Load classes
        self.classes = load_classes(self.classes_path) # Extracts class labels from file
        self.classes_colors = {}
        
        # Define subscribers
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.imageCb, queue_size = 1, buff_size = 2**24)

        # Define publishers
        self.pub_ = rospy.Publisher(self.detected_objects_topic, Detection2DArray, queue_size=10)
        self.pub_viz_ = rospy.Publisher(self.published_image_topic, Image, queue_size=10)
        rospy.loginfo("Launched node for object detection")

    def imageCb(self, data):
        # Convert the image to OpenCV
        #self.cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
        self.cv_image = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)

        # Initialize detection results
        detection_results = Detection2DArray()
        detection_results.header = data.header

        # Configure input
        input_img = self.imagePreProcessing(self.cv_image)

        # set image type
        if(torch.cuda.is_available() and self.use_cuda):
          input_img = Variable(input_img.type(torch.cuda.FloatTensor))
        else:
          input_img = Variable(input_img.type(torch.FloatTensor))

        # Get detections from network
        with torch.no_grad():
            detections = self.model(input_img)
            detections = non_max_suppression(detections, 80, self.confidence_th, self.nms_th)
        
        # Parse detections
        if detections[0] is not None:
            for detection in detections[0]:
                # Get xmin, ymin, xmax, ymax, confidence and class
                xmin, ymin, xmax, ymax, conf, _, det_class = detection
                pad_x = max(self.h - self.w, 0) * (self.network_img_size/max(self.h, self.w))
                pad_y = max(self.w - self.h, 0) * (self.network_img_size/max(self.h, self.w))
                unpad_h = self.network_img_size-pad_y
                unpad_w = self.network_img_size-pad_x
                xmin_unpad = ((xmin-pad_x//2)/unpad_w)*self.w
                xmax_unpad = ((xmax-xmin)/unpad_w)*self.w + xmin_unpad
                ymin_unpad = ((ymin-pad_y//2)/unpad_h)*self.h
                ymax_unpad = ((ymax-ymin)/unpad_h)*self.h + ymin_unpad

                # Populate darknet message
                detection_msg = Detection2D()
                detection_msg.bbox.size_x = xmax_unpad - xmin_unpad
                detection_msg.bbox.size_y = ymax_unpad - ymin_unpad
                detection_msg.bbox.center.x = (xmax_unpad + xmin_unpad)/2
                detection_msg.bbox.center.y = (ymax_unpad + ymin_unpad)/2
                result_msg = ObjectHypothesisWithPose()
                result_msg.score = conf
                result_msg.id = int(det_class)
                detection_msg.results.append(result_msg)

                # Append in overall detection message
                detection_results.detections.append(detection_msg)

            # Publish detection results
            self.pub_.publish(detection_results)

            # Visualize detection results
            if (self.publish_image):
                self.visualizeAndPublish(detection_results, self.cv_image)
        return True
    

    def imagePreProcessing(self, img):
        input_img = torch.from_numpy(np.ascontiguousarray(np.transpose(img.copy(), (2,0,1)))).float().to(self.device)
        # Extract image and shape
        height, width, channels = img.shape
        
        self.h = height
        self.w = width
        
        # Determine image to be used
        padded_image = torch.zeros((channels, max(self.h,self.w), max(self.h,self.w)), dtype=torch.float, device=self.device)
            
        # Add padding
        if (self.w > self.h):
            padded_image[:, (self.w-self.h)//2 : self.h + (self.w-self.h)//2, :] = input_img
        else:
            padded_image[:, :, (self.h-self.w)//2 : self.w + (self.h-self.w)//2] = input_img
        
        # Resize and normalize
        resized_img = torch.nn.functional.interpolate(padded_image[None], size=(self.network_img_size, self.network_img_size))/255.
        return resized_img


    def visualizeAndPublish(self, output, imgIn):
        # Copy image and visualize
        imgOut = np.ascontiguousarray(imgIn)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.6
        thickness = int(2)
        for index in range(len(output.detections)):
            label = self.classes[output.detections[index].results[0].id]
            size_x = output.detections[index].bbox.size_x
            size_y = output.detections[index].bbox.size_y
            center_x = output.detections[index].bbox.center.x
            center_y = output.detections[index].bbox.center.y

            x_p1 = center_x - size_x/2
            y_p1 = center_y - size_y/2
            x_p3 = center_x + size_x/2
            y_p3 = center_y + size_y/2
            confidence = output.detections[index].results[0].score

            # Find class color
            if label in self.classes_colors.keys():
                color = self.classes_colors[label]
            else:
                # Generate a new color if first time seen this label
                color = np.random.randint(0,255,3)
                self.classes_colors[label] = color
            
            # Create rectangle
            start_point = (int(x_p1), int(y_p1))
            end_point = (int(x_p3), int(y_p3))
            lineColor = (int(color[0]), int(color[1]), int(color[2]))

            cv2.rectangle(imgOut, start_point, end_point, lineColor, thickness)
            text = ('{:s}: {:.3f}').format(label,confidence)
            cv2.putText(imgOut, text, (int(x_p1), int(y_p1+20)), font, fontScale, (255,255,255), thickness ,cv2.LINE_AA)

        # Publish visualization image
        #image_msg = self.bridge.cv2_to_imgmsg(imgOut, "rgb8")
        image_msg = Image()
        image_msg.encoding = "bgr8"
        image_msg.height = imgOut.shape[0]
        image_msg.width = imgOut.shape[1]
        image_msg.step = image_msg.width*3
        image_msg.data = imgOut.tobytes()
        self.pub_viz_.publish(image_msg)


if __name__=="__main__":
    # Initialize node
    rospy.init_node("detector_manager_node")

    # Define detector object
    dm = DetectorManager()

    # Spin
    rospy.spin()
