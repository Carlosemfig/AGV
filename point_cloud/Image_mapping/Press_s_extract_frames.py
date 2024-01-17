import cv2
import os
import time
# Create the output folder if it doesn't exist
output_folder = 'images'
os.makedirs(output_folder, exist_ok=True)

# Open the video file
video_path = 'calib_2.avi'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Find the next available frame index
existing_frames = [int(fname.split('_')[1].split('.')[0]) for fname in os.listdir(output_folder) if fname.startswith('frame_')]
num = max(existing_frames) + 1 if existing_frames else 0


# Set the window size
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop if there are no more frames

    # Display the frame
    cv2.imshow('Frame', frame)

    time.sleep(0.2)  # Adjust the delay as needed

    # Wait for the 's' key to save the frame
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # Save the frame with the next available index
        while os.path.exists(os.path.join(output_folder, f'frame_{num}.png')):
            num += 1
        image_path = os.path.join(output_folder, f'frame_{num}.png')
        cv2.imwrite(image_path, frame)
        print(f"Frame {num} saved as {image_path}")
        num += 1

    # Exit when 'q' key is pressed
    elif key == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
