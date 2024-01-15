#intrinsic transformation
import numpy as np
import math
import cv2



#im_coordinates=[u,v,1]




image_resolution=(1280,720)
f=3.6

#fx and fy are the same value if the image is not distorted
xc=2
yc=2
zc=5
fx=f
fy=f
ox=image_resolution[0]/2
oy=image_resolution[1]/2

points_world_cam2 = np.array([[0, 0, 0,],
                         [1.25, 2.2, -0.4],
                         [1.94, 2.22, -0.18],
                         [2.65, 0.35, 0]])

points_world_cam1 = np.array([[0, 0, 0],
                        [0.53, -1.89, -0.41],
                         [2.65, 0.35, 0],
                         [1.05, -1.9, 0.2]
                         ])


points_cam2=np.array([[224, 455],
                         [757, 730],
                         [957, 658],
                         [1530, 563]])

points_cam1=np.array([[1468, 541],
                      [993, 686],
                         [156, 545],
                         [814, 513],
                         ])



from scipy.spatial.transform import Rotation
def get_intrinsic_matrix(f,image_resolution):
    fx=f
    fy=f
    ox=image_resolution[0]/2
    oy=image_resolution[1]/2
    
    matrix=np.array([[fx, 0, ox],
                [0, fy, oy],
                [0, 0, 1]])
    
    return matrix

def get_extrinsic_matrix_cam1(points_world,points_image,intrinsic_matrix):
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

def get_extrinsic_matrix_cam2(points_world,points_image,intrinsic_matrix):
    cam_matrix = np.array(intrinsic_matrix, dtype=np.float32)
    objectPoints = points_world.astype(np.float32)
    imagePoints = points_image.astype(np.float32)
    #print("IMAGE POINTS", imagePoints)
    #print("OBJECT POINTS",objectPoints)
    _,rvecs,tvecs = cv2.solvePnP(objectPoints, imagePoints, cam_matrix, distCoeffs=None, flags=cv2.SOLVEPNP_P3P  )
    #print("RVEC",rvecs)
    rotation_matrix, _ = cv2.Rodrigues(rvecs)
    extrinsic_matrix = np.hstack((rotation_matrix, tvecs))

    return extrinsic_matrix

def map_to_pixel(map_coordinates, extrinsic_matrix, cam_matrix):
    # Add a row [0, 0, 0, 1] to map_coordinates to make it a 4x1 matrix

    
    #print("extrinsic_mat",extrinsic_matrix)
    x_map, y_map, z_map=map_coordinates
    map_coordinates_4x1 = np.array([[x_map, y_map, z_map, 1]])
    # Reshape to (1, 4) if it's a 1D array
    map_coordinates_4x1 = map_coordinates_4x1.reshape((1, -1))

    #print("map_coordinates", map_coordinates_4x1)

    # Project the map coordinates to pixel coordinates
    cam_coordinates = np.dot(extrinsic_matrix, map_coordinates_4x1.T)

    #print("map_coordinates",map_coordinates_4x1)
    # Project the map coordinates to pixel coordinates
    cam_coordinates=cam_coordinates[:3]
    # Reshape to (1, 3) if it's a 1D array
    cam_coordinates = cam_coordinates.reshape((1, -1))
    #print("cam coordinates",cam_coordinates)
    #print("cam matrix",cam_matrix)
    # Apply camera intrinsic matrix
    pixel_coordinates = np.dot(cam_matrix,cam_coordinates.T)

    #print("pixel_coordinates",pixel_coordinates)
    pixel_coordinates_homogeneous = pixel_coordinates[:2] / pixel_coordinates[2]



    return pixel_coordinates_homogeneous

intrinsic_matrix_cam2=get_intrinsic_matrix(f,image_resolution)
extrinsic_matrix_cam2=get_extrinsic_matrix_cam2(points_world_cam2,points_cam2,intrinsic_matrix_cam2)


print(map_to_pixel((2.65,0.35,0),extrinsic_matrix_cam2,intrinsic_matrix_cam2))


intrinsic_matrix_cam1=get_intrinsic_matrix(f,image_resolution)
extrinsic_matrix_cam1=get_extrinsic_matrix_cam1(points_world_cam1,points_cam1,intrinsic_matrix_cam1)


print(map_to_pixel((0,0,0),extrinsic_matrix_cam1,intrinsic_matrix_cam1))