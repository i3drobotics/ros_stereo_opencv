"""
Copyright (c) 2018 I3D Robotics Ltd

Author: Josh Veitch-Michaelis
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import yaml
import numpy as np
import cv2

class roscv:
        
    """
    
    Utility class to produce rectified images and point clouds from
    ROS calibration files.
    
    Create an object by passing in the left and right yaml files
    as produced by ROS' camera calibration tool.
    
    rcv = roscv(left_file, right_file)
    
    Then you can run rcv.rectify to get rectified images, pass the output
    into a stereo matcher of your choice, and then run rcv.point_cloud to
    get 3D output.
    
    """
    
    def _load_image_size(self, path):
        with open(path, 'r') as f:
            cal = yaml.load(f)

        return (cal['image_width'], cal['image_height'])

    def _load_distortion_coefficients(self, path):
        with open(path, 'r') as f:
            cal = yaml.load(f)

        return np.array(cal['distortion_coefficients']['data'], dtype='float32').reshape(1,5)

    def _load_rectification_matrix(self, path):
        with open(path, 'r') as f:
            cal = yaml.load(f)

        return np.array(cal['rectification_matrix']['data'], dtype='float32').reshape(3,3)

    def _load_camera_matrix(self, path):
        with open(path, 'r') as f:
            cal = yaml.load(f)

        return np.array(cal['camera_matrix']['data'], dtype='float32').reshape(3,3)

    def _load_projection_matrix(self, path):
        with open(path, 'r') as f:
            cal = yaml.load(f)

        return np.array(cal['projection_matrix']['data'], dtype='float32').reshape(3,4)

    def _load_rt(self, path):
        p = self._load_projection_matrix(path)
        res = cv2.decomposeProjectionMatrix(p)
        cameraMatrix, rotMatrix, transVect = res[:3]
        R = rotMatrix.astype('float64')
        T = (transVect.astype('float64') / transVect.astype('float64')[3])[:3]
        return R, T
        
    def __init__(self, left_cal_file, right_cal_file):
        """Create a converter to process ROS camera calibration files

        Args:
            left_cal_file: left camera yaml file
            right_cal_file: right camera yaml file

        Returns:
            None
            
        """
        self.left_cameramatrix = self._load_camera_matrix(left_cal_file)
        self.right_cameramatrix = self._load_camera_matrix(right_cal_file)
        self.left_distortion = self._load_distortion_coefficients(left_cal_file)
        self.right_distortion = self._load_distortion_coefficients(right_cal_file)
        self.R, self.T = self._load_rt(right_cal_file)
        self.image_size = self._load_image_size(left_cal_file)

        self.left_rectification_matrix = self._load_rectification_matrix(left_cal_file)
        self.right_rectification_matrix = self._load_rectification_matrix(right_cal_file)
        self.left_projection_matrix = self._load_projection_matrix(left_cal_file)
        self.right_projection_matrix = self._load_projection_matrix(right_cal_file)
        
        R1, R2, P1, P2, self.Q, roi1, roi2 = cv2.stereoRectify(self.left_cameramatrix, 
                                                          np.zeros((5)), 
                                                          self.right_cameramatrix,
                                                          np.zeros((5)),
                                                          self.image_size,
                                                          self.R,
                                                          self.T)
        
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(self.left_cameramatrix,
                                                                     self.left_distortion,
                                                                     self.left_rectification_matrix,
                                                                     self.left_projection_matrix,
                                                                     self.image_size,
                                                                     cv2.CV_32FC1)
        
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(self.right_cameramatrix,
                                                                       self.right_distortion,
                                                                       self.right_rectification_matrix,
                                                                       self.right_projection_matrix,
                                                                       self.image_size,
                                                                       cv2.CV_32FC1)
        
        
    
    
    def rectify(self, left_image, right_image):
        """Rectify input left and right images.

        Args:
            left_image: path to left image
            right_image: path to right image
        
        Returns:
            left_rectified_image: left rectified image (Numpy array)
            right_rectified_image: left rectified image (Numpy array)

        """
        left = cv2.imread(left_image, cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(right_image, cv2.IMREAD_GRAYSCALE)
        
        left_rect = cv2.remap(left, self.left_map1, self.left_map2, cv2.INTER_CUBIC)
        right_rect = cv2.remap(right, self.right_map1, self.right_map2, cv2.INTER_CUBIC)
        
        return (left_rect, right_rect)
    
    def point_cloud(self, disparity):
        """Produce a point cloud from a disparity map (or list of disparities)

        Args:
            disparity: disparity map
        
        Returns:
            The output point cloud in units of metres (most likely)


        The disparity map will be converted to float32 in this function, so no
        need to explicitly cast. Remember to divide your disparities by 16 if 
        using an OpenCV2 matcher like BM or SGBM.
        """
        return cv2.reprojectImageTo3D(disparity.astype('float32'), self.Q, handleMissingValues=True).reshape((-1,3))
