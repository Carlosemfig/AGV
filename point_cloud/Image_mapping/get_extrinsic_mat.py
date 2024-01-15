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



points_world_cam2 = np.array([
                         [1.94, 2.22, -0.18],
                         [2.65, 0.35, 0],
                         [0, 0, 0,],
                         [1.25, 2.2, -0.4]
                         
                         ])




points_cam2=np.array([
                         [957, 658],
                         [1530, 563],
                         [224, 455],
                         [757, 730]
                         
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
    _,rvecs,tvecs = cv2.solvePnP(objectPoints, imagePoints, cam_matrix, distCoeffs=None, flags=cv2.SOLVEPNP_AP3P )
    #print("RVEC",rvecs)
    rotation_matrix, _ = cv2.Rodrigues(rvecs)
    extrinsic_matrix = np.hstack((rotation_matrix, tvecs))

    return extrinsic_matrix



intrinsic_matrix=cameraMatrix
extrinsic_matrix=get_extrinsic_matrix(points_world_cam2,points_cam2,intrinsic_matrix)

print("World Points",points_world_cam2)
print("Pixel Points",points_cam2)
print("Extrinsic matrix:", extrinsic_matrix)


# Save the extrinsic matrix for later use
pickle.dump(extrinsic_matrix, open("extrinsic_matrix_cam2.pkl", "wb"))


