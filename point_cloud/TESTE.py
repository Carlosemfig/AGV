import numpy as np
from scipy.spatial.transform import Rotation


def predict_next_pos(prev_pos, vector_front, v, alpha, delta_t):
    # Convert the 2D vector to a 3D vector by adding a zero for the third dimension
    vector_front_3d = np.append(vector_front, 0.0)

    # Rotate the vector_front by the given angle alpha
    rotation = Rotation.from_euler('z', alpha, degrees=True)
    rotated_vector_3d = rotation.apply(vector_front_3d)
     # Extract the rotated 2D vector from the rotated 3D vector
    rotated_vector = rotated_vector_3d[:2]
    
    # Calculate the distance traveled in the next time step
    distance = v * delta_t

    # Calculate the next position
    next_pos = prev_pos[:2] + distance * rotated_vector
    next_position = np.array([next_pos[0], next_pos[1], prev_pos[2]])

    return next_position


# Initial position [x, y, theta]
prev_pos = np.array([0.0, 0.0, 0.0])

# Velocity (speed) of the robot
v = 1.0

# Angle in degrees to rotate the front vector
alpha = 30.0

# Time step
delta_t = 1.0

# Front vector [x, y] (Assuming the front points in the positive x direction initially)
vector_front = np.array([1.0, 0.0])


result = predict_next_pos(prev_pos, vector_front, v, alpha, delta_t)

print("Initial Position:", prev_pos)
print("Initial Front Vector:", vector_front)
print("Predicted Next Position:", result)


import numpy as np
from scipy.spatial.transform import Rotation
import math
def points_to_angle(point_p,point_t):
    """
    Calculates de angle between the x axis and the p_t segment.

    Parameters:
    point_p (np.array): The center of the robot.
    point_t (np.array): The checkpoint where the robot is going.

    Returns:
    angle_deg (float): Returns the angle in degrees that corresponds to x axis and the p_t segment angle.
    """

    x_t=point_t[0]
    y_t=point_t[1]
    x_p=point_p[0]
    y_p=point_p[1]
    angle_radians = math.atan2((y_t - y_p),(x_t - x_p))
    angle_degrees = math.degrees(angle_radians)
    angle_deg=round(angle_degrees,2)
    return angle_deg


def calculate_velocity_and_alpha(prev_pos, next_pos, delta_t, prev_vector_front, next_vector_front):
    displacement = next_pos[:2] - prev_pos[:2]
    distance = np.linalg.norm(displacement)
    velocity = distance / delta_t

    # Calculate the rotation angle (alpha)
    prev_front = prev_vector_front  / np.linalg.norm(prev_vector_front)
    next_front = next_vector_front / np.linalg.norm(next_vector_front)
    # Calculate the rotation axis

    prev_angle=points_to_angle((0,0),prev_front)
    print("prev_angle",prev_angle)
    next_angle=points_to_angle((0,0),next_front)
    print("next_angle",next_angle)
    alpha=next_angle-prev_angle
    
    return velocity, alpha

# Example usage with negative alpha
prev_pos = np.array([0.0, 0.0, 0.0])
next_pos = np.array([1.0, 1.0, 0.0])
delta_t = 1.0
prev_vector_front = np.array([0.0, -1.0])
next_vector_front = np.array([1.0, 0.0])  # Ensure a negative rotation angle

velocity, alpha = calculate_velocity_and_alpha(prev_pos, next_pos, delta_t, prev_vector_front, next_vector_front)

print("Velocity:", velocity)
print("Alpha:", alpha)
