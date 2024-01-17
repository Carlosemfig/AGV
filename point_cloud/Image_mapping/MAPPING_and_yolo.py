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

with open("extrinsic_matrix_cam2.pkl", "rb") as file:
    extrinsic_matrix = pickle.load(file)

# Now you can use cameraMatrix and dist in your code
print("Loaded Intrinsic Matrix:", intrinsic_matrix)
print("Loaded dist:", dist)
print("Loaded Extrinsic Matrix",extrinsic_matrix)



image_resolution=(1080, 1920)
#image_resolution=(640,360)


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

def calculate_cube_vertices(center, side_length,extrinsic_matrix,intrinsic_matrix):
    """
    This function should return the value of the pixels corresponding to the vertices of a cube:
    Centered in center and 
    with side lenght= side_lenght
    
    
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

    pixel_coordinates_array = []
    for vertex in vertices:
        # Map each vertex to pixel coordinates for camera 2
        pixel_coordinates = map_to_pixel(vertex, extrinsic_matrix, intrinsic_matrix)
        pixel_coordinates_array.append(pixel_coordinates)

    return pixel_coordinates_array


Lidar_1=map_to_pixel((0,0,0),extrinsic_matrix,intrinsic_matrix)
Lidar_2 = map_to_pixel((1.25, 2.22, -0.4), extrinsic_matrix, intrinsic_matrix)
Camera_1 = map_to_pixel((1.94, 2.22, -0.18), extrinsic_matrix, intrinsic_matrix)
Random_object = map_to_pixel((2.65, 0.35, 0), extrinsic_matrix, intrinsic_matrix)


center = (0, 0, 0)
side_length = 0.2
pixel_coordinates_array = calculate_cube_vertices(center, side_length,extrinsic_matrix,intrinsic_matrix)


# Load the image to visualize the resulting pixels
#in red the known points
#in green the cube points
image_path = "output_frame_2.jpg"
output_frame = cv2.imread(image_path)
output_frame_copy = output_frame.copy()


# Draw green points on the image
for pixel_coordinates in pixel_coordinates_array:
    # Round to integers as pixel coordinates must be integers
    x, y = map(int, pixel_coordinates)
    cv2.circle(output_frame, (x, y), 5, (0, 255, 0), -1)  # Green circle with radius 5

#Draw the red points in the image
red_circles = [Lidar_1, Lidar_2, Camera_1, Random_object]
for pixel_coordinates in red_circles:
    # Round to integers as pixel coordinates must be integers
    x, y = map(int, pixel_coordinates)
    cv2.circle(output_frame, (x, y), 5, (0, 0, 255), -1)  # Red circle with radius 5


def find_min_max_coordinates(points):
    min_x = int(np.min([point[0] for point in points]))
    max_x = int(np.max([point[0] for point in points]))
    min_y = int(np.min([point[1] for point in points]))
    max_y = int(np.max([point[1] for point in points]))
    return min_x, max_x, min_y, max_y

# Draw a green rectangle on the image
min_x, max_x, min_y, max_y = find_min_max_coordinates(pixel_coordinates_array)
cv2.rectangle(output_frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), thickness=2)



# Extract the region inside the rectangle
rectangle_segmentation = output_frame_copy[min_y:max_y, min_x:max_x]
#cv2.imwrite('Rectangle_segmentation.png', rectangle_segmentation)

import cv2
import argparse
import numpy as np

def yolo_object_detection(image_path, config, weights, classes_file):
    image=image_path

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(weights, config)

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    def get_output_layers(net):
        layer_names = net.getLayerNames()
        try:
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        except:
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(classes[class_id])
        color = COLORS[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        try:
            box = boxes[i]
        except:
            i = i[0]
            box = boxes[i]

        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

    return image

# Example usage:

config = r"C:\Users\hvendas\Desktop\YOLO_V3\object-detection-opencv-master\yolov3.cfg"
weights = r"C:\Users\hvendas\Desktop\YOLO_V3\object-detection-opencv-master\yolov3.weights"
classes_file = r"C:\Users\hvendas\Desktop\YOLO_V3\object-detection-opencv-master\yolov3.txt"

result_image = yolo_object_detection(rectangle_segmentation, config, weights, classes_file)

# Display or save the result
cv2.imshow("Object Detection on Rectangle Segmentation", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()





# Display the image with drawn points
cv2.imshow("Output Frame", output_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()