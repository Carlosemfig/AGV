import point_cloud_pb2
import sys
import numpy as np
import MAIN as m



point_cloud_messages = point_cloud_pb2.PointCloud()
# Open the binary file in binary mode
with open('serialized_data.bin', 'rb') as file:
    # Read the binary data
    binary_data = file.read()

# Create an instance of the protobuf message
#point_cloud_messages = point_cloud_pb2.Point()  # Replace with your actual message name

# Parse the binary data into the protobuf message
point_cloud_messages.ParseFromString(binary_data)
print(point_cloud_messages)
coordinates=[]

for point in point_cloud_messages.points:
    x = point.x
    y = point.y
    z = point.z
    
    # Append the coordinates to the list
    coordinates.append([x, y, z])

# Convert the list of coordinates to a NumPy array for easier manipulation
coordinates_array = np.array(coordinates)
print(coordinates_array)

result=m.array_to_pc(coordinates_array)

m.visualize(result)


"""
def decode_point_cloud(file_path):
    with open(file_path, "rb") as f:
        deserialized_data = f.read()
    deserialized_point_cloud = point_cloud_pb2.PointCloud()
    deserialized_point_cloud.ParseFromString(deserialized_data)
    return deserialized_point_cloud"""
"""
def main():
    if len(sys.argv) < 2:
        print("Usage: python decode_point_cloud.py <file_path>")
    else:
        file_path = sys.argv[1]
        decoded_point_cloud = decode_point_cloud(file_path)
        total_points = len(decoded_point_cloud.points)
        timestamp = decoded_point_cloud.timestamp
        print(f"Total points: {total_points}, Timestamp: {timestamp}")
 
if __name__ == "__main__":
    main()"""