# ros_stereo_opencv

A tool for working with stereo images in OpenCV using ROS calibration files

This repository contains a Python class which may be useful for dealing with stereo images that have been captured in ROS.

**Dependencies:** cv2 (built from OpenCV3), yaml, numpy

This library was tested using Python 3.6, but should be easily convertable to 2.7 or older versions of OpenCV.

# Usage

Here we define a couple of convenience functions to process a stereo pair and output the point cloud as a PLY file:

```
import cv2
import numpy as np
from roscv import roscv

def write_ply(fn, verts, intensity):
    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        '''

    verts = np.hstack([verts, intensity])

    with open(fn, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))

    with open(fn, 'ab') as f:
        np.savetxt(f, verts, '%f %f %f %d %d %d')
        
def process_stereopair(rcv, matcher, left, right, out, thresh=1):
    left_rect, right_rect = rcv.rectify(left, right)
    disparity = matcher.compute(left_rect, right_rect).astype('float32')/16.0
    
    # Clean up the disparity map a little
    disparity = cv2.medianBlur(disparity, 3)
        
    points = rcv.point_cloud(disparity)
    mask = abs(points[:,2]) <= thresh
    points[mask]
    
    colours = np.tile(left_rect.reshape((-1,1)), 3)
    write_ply(out, points[mask], colours[mask])


left_cal_file = './camera_info/left.yaml'
right_cal_file = './camera_info/right.yaml'

rcv = roscv(left_cal_file, right_cal_file)

matcher = cv2.StereoSGBM_create(0, 64, 15)
matcher.setSpeckleRange(100)
matcher.setSpeckleWindowSize(50)

process_stereopair(rcv, matcher, './left.png', './right.png', 'output.ply')
```

It boils down to creating a conversion object using your ROS cal files, rectifying the images, matching them and then using the output disparity map to produce a point cloud.

# Implementation Details
By convention, ROS stores camera calibration information in yaml files that contain the camera matrix, distortion coefficients, rectification matrix and projection matrix. This is minimally sufficient information to rectify and produce point clouds from stereo images captured in ROS, but requires a number of steps in OpenCV before it's actually usable.

Here is an example from i3Dr's Deimos camera:

```
image_width: 752
image_height: 480
camera_name: cameraRight
camera_matrix:
  rows: 3
  cols: 3
  data: [722.634460, 0.000000, 377.000978, 0.000000, 720.197898, 248.585059, 0.000000, 0.000000, 1.000000]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [0.075413, -0.123377, 0.001762, -0.000622, 0.000000]
rectification_matrix:
  rows: 3
  cols: 3
  data: [0.999745, -0.000276, -0.022582, 0.000236, 0.999998, -0.001740, 0.022583, 0.001734, 0.999743]
projection_matrix:
  rows: 3
  cols: 4
  data: [748.174074, 0.000000, 392.540657, -44.487442, 0.000000, 748.174074, 246.805559, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000]
```
This package first loads the various matrices and attemps to produce rectification maps for each camera. For some reason, ROS does not store the camera intrinsic geometry directly in the files (i.e. the rotation and translation between the cameras), so we have to extract it from the projection matrix.

The parameters provided are a mix of unrectified and rectified. For instance, the camera matrix and distortion coefficients represent the real, unrectified, camera (i.e. what you'd get from a monocular calibration). The rectification and projection matrices encode information about the rectified stereo system.

We use a mix to get the rectification maps. The rectification and projection matrices can be passed to `initUndistortRectifyMap` along with the distortion coefficients and camera matrix. This doesn't get us Q, however, the matrix used to project from disparity and 3D. For that we need R and T.

In order to get R and T, we can decompose the projection matrix. Recall that this R and T is the _rectified_ transform, so both cameras have zero rotation and a single, horizontal, translation. We also need to use the rectified camera matrices (gotten from the projection matrix) and zero distortion coefficients. With that, we can produce Q.

The rest is straightforward OpenCV. We produce the rectification maps (using bicubic interpolation) and provide a function to rectify input images. We also provide a function to take in a disparity map (or a list of disparities) and output a point cloud.




