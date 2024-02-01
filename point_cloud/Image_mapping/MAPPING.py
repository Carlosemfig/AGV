#intrinsic transformation
import numpy as np
import math
import cv2
import pickle

# Load cameraMatrix from file
with open("cameraMatrix.pkl", "rb") as file:
    intrinsic_matrix = pickle.load(file)

# Load dist from file
with open("dist.pkl", "rb") as file:
    dist = pickle.load(file)

with open("extrinsic_matrix_cam1.pkl", "rb") as file:
    extrinsic_matrix = pickle.load(file)

# Now you can use cameraMatrix and dist in your code
print("Loaded Intrinsic Matrix:", intrinsic_matrix)
print("Loaded dist:", dist)
print("Loaded Extrinsic Matrix",extrinsic_matrix)



image_resolution=(1080, 1920)
#image_resolution=(640,360)

Lidar_1=[0, 0, 0]
Lidar_2=[1.29, 1.07, 0]
Lidar_3=[1.99, -0.75, 0]
 
Cam_1=[0, 0.89, -0.17]
Cam_2=[0, -0.81, -0.17]
Cam_3=[2.55, 0.95, -0.17]
 
Peluche=[2.31, 0, -0.10]




center = Cam_3
side_length = 0.2




def map_to_pixel(map_coordinates, extrinsic_matrix, cam_matrix):
    """
    Comverts world coordinates to pixel_coordinates.

    Param:
    Map_coordinates(x,y,z): The spacial coordinates of one of the coorners of the bounding boxes
    Extrinsic_matrix(np.array):Resulting from the camera calibration
    Intrinsic_matrix(np_array):Resulting from the camera calibration
    

    Returns:
    Pixel_coordinates_homogeneus(x,y): The coordinates of the pixel that corresponds to the spacial point given as the map_coordinates.
    """
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





first=map_to_pixel(Lidar_2,extrinsic_matrix,intrinsic_matrix)
second = map_to_pixel(Cam_3, extrinsic_matrix, intrinsic_matrix)
third = map_to_pixel(Lidar_3, extrinsic_matrix, intrinsic_matrix)
fourth = map_to_pixel(Peluche, extrinsic_matrix, intrinsic_matrix)



def get_cube_vertices(center, side_length):
    """
    Returns the vertices coordinates of a cube with given coordinates. This function is being used for visualization purposes.

    Param
    center(x,y,z) the spacial corrdinates of the cube center
    side_lenght (float) ist the lenght of the side of the cube
    
    Returns
    vertices(np.array) - with the 8 vertices of the cube in the (x,y,z) format.
    
    
    """
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

vertices=get_cube_vertices(center,side_length)

def vertices_to_pixels(vertices,extrinsic_matrix,intrinsic_matrix):
    """
    Transforms the spacial coordinates of the vetices points to pixel coordinates in the image.

    Param
    vertices(np.array) an array with the box vertices in spacial coordinates(x,y,z)
    Extrinsic_matrix(np.array):Resulting from the camera calibration
    Intrinsic_matrix(np_array):Resulting from the camera calibration

    returns:
    pixel_coordinates_array(list): is a list containing the pixel coordinates (x,y) for each given vertice (8 values).
    
    
    """
    pixel_coordinates_array = []
    for vertex in vertices:
        # Map each vertex to pixel coordinates for camera 2
        pixel_coordinates = map_to_pixel(vertex, extrinsic_matrix, intrinsic_matrix)
        pixel_coordinates_array.append(pixel_coordinates)

    return pixel_coordinates_array


pixel_coordinates_array = vertices_to_pixels(vertices,extrinsic_matrix,intrinsic_matrix)


# Load the image to visualize the resulting pixels
#in red the known points
#in green the cube points
image_path = "cam_1_extrinsic.jpg"
output_frame = cv2.imread(image_path)
output_frame_copy = cv2.imread(image_path)
print(type(output_frame))


# Draw green points on the image
for pixel_coordinates in pixel_coordinates_array:
    # Round to integers as pixel coordinates must be integers
    x, y = map(int, pixel_coordinates)
    cv2.circle(output_frame, (x, y), 5, (0, 255, 0), -1)  # Green circle with radius 5

#Draw the red points in the image
red_circles = [first, second, third, fourth]
for pixel_coordinates in red_circles:
    # Round to integers as pixel coordinates must be integers
    x, y = map(int, pixel_coordinates)
    print("COORDENADAS",x,y)
    cv2.circle(output_frame, (x, y), 5, (0, 0, 255), -1)  # Red circle with radius 5


def find_min_max_coordinates(points):
    """
    Funstion to find the min and max bound in both direction to include all the pixels that corresponds to the vertices of the box.
    Param:
    points(list): is a list containing the pixel coordinates (x,y) for each given vertice (8 values).

    Returns:
    min_x, max_x, min_y, max_y(int): the min and max bounds in both direction.
    
    """
    min_x = int(np.min([point[0] for point in points]))
    max_x = int(np.max([point[0] for point in points]))
    min_y = int(np.min([point[1] for point in points]))
    max_y = int(np.max([point[1] for point in points]))
    return min_x, max_x, min_y, max_y


min_x, max_x, min_y, max_y = find_min_max_coordinates(pixel_coordinates_array)

def segment_image(image, min_x, max_x, min_y, max_y):
    # Check if the image is not None
    if image is not None:
        # Ensure none of the values is equal to 0
        min_x = max(1, min_x)
        min_y = max(1, min_y)

        # Ensure max values are within the image dimensions
        max_x = min(max_x, image.shape[1] - 1)
        max_y = min(max_y, image.shape[0] - 1)

        # Crop the image
        rectangle_segmentation = image[min_y:max_y, min_x:max_x]

        return rectangle_segmentation





print("coordinates",min_x,max_x,min_y,max_y)
# Extract the region inside the rectangle
segmented_image=segment_image(output_frame_copy,min_x,max_x,min_y,max_y)
cv2.imwrite('Rectangle_segmentation.png', segmented_image)


cv2.rectangle(output_frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), thickness=2)
# Display the image with drawn points
# Resize the image with drawn points
resized_frame = cv2.resize(output_frame, (1200, 675))  # Replace new_width and new_height with your desired dimensions

cv2.imshow("Output Frame", resized_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()