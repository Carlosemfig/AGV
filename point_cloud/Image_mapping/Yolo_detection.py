import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import pickle

import torch
from pathlib import Path
from torchvision.transforms import functional as F
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import time_synchronized
from utils.datasets import letterbox

def check_img_size(img_size, s=32, min=32, stride=32):
    """
    Auxiliar function. Needs to be called.
    """
    img_size = max(min, int(img_size / stride) * stride)
    return img_size


def detect_single_image(image, weights,conf_thres = 0.5):
    """
    Function that recives an image and the weights of the model, and returns the bbox coordinates of the 2dimensional box detected.

    Param:
    image(np.array): The image resulting from the segmentation process.
    Weights(file.pt): The path to the weights of the model. This file will be updated allong the project.
    conf_thresh(float): The bigger this number more strict we are with the object we want to find.

    Returns:
    bbox_info(list): Contains the information regarding the objects detected in the image. Category, bbox coordinates and confidence.
    """

    img_size = 640  # You can adjust this based on your preference
      # Confidence threshold for detections
    iou_thres = 0.45  # IOU threshold for non-maximum suppression

    # Load model
    model = attempt_load(weights, map_location=torch.device('cpu'))
    stride = int(model.stride.max())
    imgsz = check_img_size(img_size, s=stride)

    # Preprocess the input image
    img = letterbox(image, new_shape=imgsz)[0]
    img = F.to_tensor(img).unsqueeze(0)

    # Run inference
    t0 = time_synchronized()
    with torch.no_grad():
        pred = model(img)[0]
    t1 = time_synchronized()

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    # Process detections
    if len(pred):
        # Rescale boxes to original image size
        det = pred[0]
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()

        # Create a list to store bounding box information
        bbox_info = []

        for *xyxy, conf, cls in det:
            bbox_info.append({'class': int(cls), 'xyxy': [coord.item() for coord in xyxy], 'confidence': conf.item()})
    else:
        bbox_info = []  # No detections

    # Print time taken for inference
    print(f'Done. ({(1E3 * (t1 - t0)):.1f}ms) Inference')

    return bbox_info


# Load the image
image_path = 'TEST_IMAGE.jpeg'
image = cv2.imread(image_path)

# Load the weights
weights_path = 'Best_v2.pt'

# Call the detection function
bbox_info = detect_single_image(image, weights_path)



# Print bounding box information
print("Bounding Box Information:")
for bbox in bbox_info:
    print(bbox)



def draw_bounding_boxes(image, bbox_info):

    """
    Receives the output of the previous function and the image and draws the bboxes so  that we can visualize it.

    Returns:
    Image(np.array)
    """
    result_image = image.copy()

    for bbox in bbox_info:
        class_id = bbox['class']
        xywh = bbox['xyxy']
        confidence = bbox['confidence']

        # Extract coordinates
        x, y, w, h = xywh

        # Draw bounding box
        color = (0, 255, 0)  # Green color (you can change it as needed)
        thickness = 2
        result_image = cv2.rectangle(result_image, (int(x), int(y)), (int(w), int(h)), color, thickness)

        # Add label with confidence
        label = f'Class {class_id}: {confidence:.2f}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        text_origin = (int(x), int(y - 5))
        result_image = cv2.putText(result_image, label, text_origin, font, font_scale, color, font_thickness)

    return result_image


final_image = draw_bounding_boxes(image, bbox_info)
cv2.imshow('Result Image with Bounding Boxes', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()