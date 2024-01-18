#intrinsic transformation
import numpy as np
import math
import cv2
import pickle

# Load cameraMatrix from file
with open("cameraMatrix.pkl", "rb") as file:
    cameraMatrix = pickle.load(file)

# Load dist from file
with open("dist.pkl", "rb") as file:
    dist = pickle.load(file)

# Now you can use cameraMatrix and dist in your code
print("Loaded cameraMatrix:", cameraMatrix)
print("Loaded dist:", dist)





#im_coordinates=[u,v,1]

image_resolution=(1080, 1920)


pixel_cam2=np.array([
                         [405, 607],
                         [1726, 523],
                         [1198, 685],
                         [891, 705]
                         
                         
                         
                         ])
pixel_cam3=np.array([
                         [186, 544],
                         [973, 588],
                         [1297, 502],
                         [1698, 600],
                         [1836, 405]
                         ])

Lidar_1=[0, 0, 0]
Lidar_2=[1.38, 1.11, 0]
Lidar_3=[2.05, -0.95, 0]
 
Cam_1=[0.05, 0.89, -0.17]
Cam_2=[-0.03, -0.81, -0.17]
Cam_3=[2.80, 0.95, -0.17]
 
Peluche=[2.31, -0.29, -0.10]

world_cam1=np.array([
                         Lidar_2,

                         Lidar_3,
                         Cam_3,
                         Peluche,
                         ])
pixel_cam1=np.array([
                         [225, 335],
                         
                         [1378, 397],
                         [364, 528],
                         [895, 528]
                         
                         
                         
                         
                         ])
world_cam2=np.array([
                         Lidar_2,
                         Lidar_3,
                         Peluche,
                         Cam_3
                         
                         
                         ])

world_cam3=np.array([
                         Peluche,
                         Cam_2,
                         Lidar_1,
                         Cam_1,
                         Lidar_2
                         ])




"""from scipy.spatial.transform import Rotation
def get_intrinsic_matrix(f,image_resolution):
    fx=f
    fy=f
    ox=image_resolution[0]/2
    oy=image_resolution[1]/2
    
    matrix=np.array([[fx, 0, ox],
                [0, fy, oy],
                [0, 0, 1]])
    
    return matrix

intrinsic_matrix_cam2=get_intrinsic_matrix(f,image_resolution)
"""
coef_distortion=dist
def get_extrinsic_matrix(points_world,points_image,intrinsic_matrix):
    cam_matrix = np.array(intrinsic_matrix, dtype=np.float32)
    objectPoints = points_world.astype(np.float32)
    imagePoints = points_image.astype(np.float32)
    #print("IMAGE POINTS", imagePoints)
    #print("OBJECT POINTS",objectPoints)
    _,rvecs,tvecs = cv2.solvePnP(objectPoints, imagePoints, cam_matrix, distCoeffs=None, flags=cv2.SOLVEPNP_P3P )
    #print("RVEC",rvecs)
    rotation_matrix, _ = cv2.Rodrigues(rvecs)
    extrinsic_matrix = np.hstack((rotation_matrix, tvecs))

    return extrinsic_matrix



intrinsic_matrix=cameraMatrix
extrinsic_matrix=get_extrinsic_matrix(world_cam1,pixel_cam1,intrinsic_matrix)

print("World Points",world_cam2)
print("Pixel Points",pixel_cam2)
print("Extrinsic matrix:", extrinsic_matrix)


# Save the extrinsic matrix for later use
pickle.dump(extrinsic_matrix, open("extrinsic_matrix_cam1.pkl", "wb"))


