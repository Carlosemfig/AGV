#intrinsic transformation
import numpy as np
import math
import cv2

#im_coordinates=[u,v,1]

image_resolution=(1080, 1920)
#image_resolution=(640,360)
f=500

#fx and fy are the same value if the image is not distorted
xc=2
yc=2
zc=5
fx=f
fy=f
ox=image_resolution[0]/2
oy=image_resolution[1]/2

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


def get_extrinsic_matrix_cam2(points_world,points_image,intrinsic_matrix):
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


print(map_to_pixel((0,0,0),extrinsic_matrix_cam2,intrinsic_matrix_cam2))


import numpy as np

def calculate_cube_vertices(center, side_length):
    x_c, y_c, z_c = center
    h = side_length / 2.0  # Half of the side length

    vertices = np.array([
        [x_c - h, y_c - h, z_c - h],  # Vertex 0
        [x_c + h, y_c - h, z_c - h],  # Vertex 1
        [x_c + h, y_c + h, z_c - h],  # Vertex 2
        [x_c - h, y_c + h, z_c - h],  # Vertex 3
        [x_c - h, y_c - h, z_c + h],  # Vertex 4
        [x_c + h, y_c - h, z_c + h],  # Vertex 5
        [x_c + h, y_c + h, z_c + h],  # Vertex 6
        [x_c - h, y_c + h, z_c + h]   # Vertex 7
    ])

    return vertices

# Example usage:
#center = (0, 0, 0)
#center = (2.65, 0.35, 0)
center = (1.94, 2.22, -0.18)
side_length = 0.2
cube_vertices = calculate_cube_vertices(center, side_length)
print(cube_vertices)


pixel_coordinates_array = []

for vertex in cube_vertices:
    # Map each vertex to pixel coordinates for camera 2
    pixel_coordinates = map_to_pixel(vertex, extrinsic_matrix_cam2, intrinsic_matrix_cam2)
    pixel_coordinates_array.append(pixel_coordinates)

# After the loop, pixel_coordinates_array contains the pixel coordinates for each vertex
print("Pixel Coordinates Array:", pixel_coordinates_array)


# Calculate the resulting pixel for the center of the cube
center_pixel_1 = map_to_pixel((0, 0, 0), extrinsic_matrix_cam2, intrinsic_matrix_cam2)
center_pixel_2 = map_to_pixel((1.25, 2.22, -0.4), extrinsic_matrix_cam2, intrinsic_matrix_cam2)
center_pixel_3 = map_to_pixel((1.94, 2.22, -0.18), extrinsic_matrix_cam2, intrinsic_matrix_cam2)
center_pixel_4 = map_to_pixel((2.65, 0.35, 0), extrinsic_matrix_cam2, intrinsic_matrix_cam2)

# Load the image
image_path = "output_frame_2.jpg"
output_frame = cv2.imread(image_path)
print("SHAPE",output_frame.shape)

# Draw green points on the image
for pixel_coordinates in pixel_coordinates_array:
    # Round to integers as pixel coordinates must be integers
    x, y = map(int, pixel_coordinates)
    cv2.circle(output_frame, (x, y), 5, (0, 255, 0), -1)  # Green circle with radius 5


red_circles = [center_pixel_1, center_pixel_2, center_pixel_3, center_pixel_4]
print("red_circles",red_circles)
for pixel_coordinates in red_circles:
    # Round to integers as pixel coordinates must be integers
    x, y = map(int, pixel_coordinates)
    cv2.circle(output_frame, (x, y), 5, (0, 0, 255), -1)  # Red circle with radius 5

# Display the image with drawn points
cv2.imshow("Output Frame", output_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()