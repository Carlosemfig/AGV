import cv2

variable=1.5
def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel_value = img[y, x]
        print(f"Clicked at coordinates (x={variable*x}, y={variable*y}). Pixel value: {pixel_value}")

# Load the image
img = cv2.imread("cam_1_extrinsic.jpg")

# Check if the image is loaded successfully
if img is None:
    print("Error: Could not load the image.")
else:
    # Get the screen resolution
    screen_resolution = (int(1920/variable),int(1080/variable))  # Replace with your actual screen resolution
    print(screen_resolution)
    # Resize the image to fit within the screen resolution
    max_width, max_height = screen_resolution
    if img.shape[1] > max_width or img.shape[0] > max_height:
        img = cv2.resize(img, (max_width, max_height))

    cv2.imshow("Image Viewer", img)

    # Set the mouse callback function
    cv2.setMouseCallback("Image Viewer", mouse_click)

    # Wait for a key press and close the window when a key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()
