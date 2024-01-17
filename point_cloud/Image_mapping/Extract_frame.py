import cv2

def extract_frame(video_path, frame_number, output_path):
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    print("Entered")
    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Set the frame position to the desired frame number
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)

    # Read the frame
    ret, frame = cap.read()

    # Check if the frame is read successfully
    if not ret:
        print("Error: Could not read frame.")
        return

    # Save the frame to the output path
    cv2.imwrite(output_path, frame)

    # Release the video capture object
    cap.release()

    print(f"Frame {frame_number} extracted and saved to {output_path}.")

# Specify the input video file, frame number, and output path
video_path = r"cam_3_extrinsic.avi"
frame_number = 2
output_path = "cam_3_extrinsic.jpg"

# Call the function to extract the 100th frame
extract_frame(video_path, frame_number, output_path)
