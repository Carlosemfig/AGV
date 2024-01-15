import cv2

def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel_value = img[y, x]
        print(f"Clicked at coordinates (x={x}, y={y}). Pixel value: {pixel_value}")

# Load the image
img = cv2.imread("output_frame_2.jpg")

# Check if the image is loaded successfully
if img is None:
    print("Error: Could not load the image.")
else:

    height, width, channels = img.shape
    print(f"Image Resolution: {width} x {height}")
    # Create a window and display the image
    cv2.imshow("Image Viewer", img)

    # Set the mouse callback function
    cv2.setMouseCallback("Image Viewer", mouse_click)

    # Wait for a key press and close the window when a key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()
