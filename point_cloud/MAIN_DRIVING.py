import MAIN as m
import numpy as np

"""____Funções relativas à condução simples(cálculo da nova posição)_____"""

def ajust_angle(beta, teta_e, delta_t):
    """
    Gives the ajustment angle that the robot need to rotate in a time step.
    Parameters:
    beta (float): Is the constant of porporcionality for the angular velocity to be calculated.
    teta_e (float): The ajustment angle, required to achive the checkpoint in degrees.
    delta_t (float): The time corresponding to a time_step

    Returns:
    teta_ajust (float): Is the angle in degrees that the robot needs to rotate in the next iteration.
    """
    ang_vel=beta*teta_e
    teta_ajust=ang_vel*delta_t
    return(teta_ajust)

def get_teta_e(bbox, checkpoints):
    """
    Calculates the orientation error angle (teta_e) while updates the rotation matrix of the bounding box.

    Parameters:
    bbox (o3d.geometry.OrientedBoundingBox()): Representation of an object in 3D space.
    checkpoints (np.array): Equally spaced checkpoints of the path.

    Returns:
    teta_e (float): The orientation error angle (in degrees) that the robot needs to rotate to face the next checkpoint.
    updated_rot_mat (np.array): The updated rotation matrix associated with the bounding box.
    """

    # Get information from the bounding box
    center = bbox.get_center()
    trajectory = bbox.get_trajectory()
    rot_mat = bbox.get_rotation_matrix()

    # Gives the direction the object was moving in the last iteration.
    vector_front = m.create_vector(trajectory[-1], trajectory[-2])

    # Finds the closest checkpoint to the center of the bounding box.
    next_checkpoint = m.find_closest_checkpoint_new(center, checkpoints)

    # Update the rotation matrix based on the object's movement direction.
    # Only if the x-axis in the object coordinates is in the contrary direction.
    updated_rot_mat = m.change_mat_bbox(rot_mat, vector_front)
    bbox.update_rotation_matrix(updated_rot_mat)

    # Calculate the angle between the x-axis of the bounding box and the x-axis in inertial coordinates.
    teta_p = m.matrix_to_angle(updated_rot_mat)
    teta_p = m.normalize_angle(teta_p)

    # Calculate the angle between the vector from the center to the next checkpoint and the x-axis in inertial coordinates.
    teta_t = m.points_to_angle(center, next_checkpoint)
    teta_t = m.normalize_angle(teta_t)

    # Calculate the rotation that the robot needs to perform to be facing the next checkpoint.
    teta_e = teta_t - teta_p

    return teta_e

def next_point_first_time(bbox,teta_e,delta_t,v_max,beta):
  rot_mat=bbox.get_rotation_matrix()
  center=bbox.get_center()
  """
    Calculates the exact position and rotation angle of the robot in the next time step.

    Parameters:
    bbox (o3d.geometry.OrientedBoundingBox()): Representation of an object in 3D space.
    teta_e (float): The orientation error angle (in degrees) that the robot is rotating to face the next checkpoint.
    delta_t (float): The time step.
    v_max (float): The maximum velocity that the robot can have.
    beta (float): Constant used to calculate the adjustment angle for each iteration.

    Returns:
    next_pos (np.array): (x, y, z) position of the robot center in the next iteration.
    teta_adjust (float): The angle (in degrees) that the robot is rotating in the next iteration.
    """
  # Get information from the bounding box
  rot_mat=bbox.get_rotation_matrix()
  center=bbox.get_center()

  #Calculate the angles that the robot is rotating in the next iteration.
  teta_ajust=ajust_angle(beta,teta_e,delta_t)
  vector_p=np.dot(rot_mat, np.array([1,0,0])) [:2]
  adj= np.radians(teta_ajust)
  adjustment_matrix = np.array([[np.cos(adj), -np.sin(adj)],
                           [np.sin(adj), np.cos(adj)]])
  
  #Apply the value of this rotation to the x_axis of the boinding box to understand the moving direction.
  pointing_direction = np.dot(adjustment_matrix, vector_p)

  #Calculation of the velocity value based on the teta_e value.
  v=m.velocity_value(teta_e,v_max)

  #Calculates the next position based on the center, the velocity and the pointing direction.
  next_pos=m.dist_to_pos(center,v*delta_t,pointing_direction)
  
  return next_pos,teta_ajust

def next_point_other_times(bbox, old_teta_e, teta_e, delta_t, v_max, beta):
    """
    Calculates the exact position and rotation angle of the robot in the next time step for subsequent iterations.

    Parameters:
    bbox (o3d.geometry.OrientedBoundingBox()): Representation of an object in 3D space.
    old_teta_e (float): The previous orientation error angle (in degrees) that the robot rotated to face the previous checkpoint.
    teta_e (float): The new orientation error angle (in degrees) that the robot is rotating to face the next checkpoint.
    delta_t (float): The time step.
    v_max (float): The maximum velocity that the robot can have.
    beta (float): Constant used to calculate the adjustment angle for each iteration.

    Returns:
    next_pos (np.array): (x, y, z) position of the robot center in the next iteration.
    teta_adjust (float): The angle (in degrees) that the robot is rotating in the next iteration.
    """

    rot_mat = bbox.get_rotation_matrix()
    center = bbox.get_center()

    # Calculate the angles that the robot is rotating in the next iteration.
    teta_adjust = ajust_angle(beta, teta_e, delta_t)
    vector_p = np.dot(rot_mat, np.array([1, 0, 0]))[:2]

    # Calculate the adjustment angle based on the previous orientation error angle.
    old_teta_adjust = ajust_angle(beta, old_teta_e, delta_t)
    adj = np.radians(old_teta_adjust)
    adjustment_matrix = np.array([[np.cos(adj), -np.sin(adj)],
                                  [np.sin(adj), np.cos(adj)]])

    # Apply the value of this rotation to the x-axis of the bounding box to understand the moving direction.
    pointing_direction = np.dot(adjustment_matrix, vector_p)

    # Calculate the velocity value based on the teta_e value.
    v = m.velocity_value(teta_e, v_max)

    # Calculates the next position based on the center, the velocity, and the pointing direction.
    next_pos = m.dist_to_pos(center, v * delta_t, pointing_direction)

    return next_pos, teta_adjust

#Estas são importadas
def first_time(bbox, checkpoints, v_max, delta_t, beta):
    """
    Initializes the orientation of the robot based on the first checkpoint.

    Parameters:
    bbox (o3d.geometry.OrientedBoundingBox()): Representation of an object in 3D space.
    checkpoints (np.array): Equally spaced checkpoints of the path.
    v_max (float): The maximum velocity that the robot can have.
    delta_t (float): The time step.
    beta (float): Constant used to calculate the adjustment angle for each iteration.

    Returns:
    Returns:
    position (np.array): (x, y, z) position of the robot center in the next iteration.
    calc_angle (float): The angle (in degrees) that the robot is rotating in the next iteration.
    """

    center = bbox.get_center()
    rot_mat = bbox.get_rotation_matrix()

    # Find the closest checkpoint to the center of the bounding box.
    next_checkpoint = m.find_closest_checkpoint_new(center, checkpoints)

    # Calculate the front vector from the center to the next checkpoint.
    vector_front = m.create_vector(next_checkpoint, center)[:2]
    # Add a zero at the end to represent the third dimension.
    vector_front = np.append(vector_front, 0)

    # The rot_mat of the bbox may need to be updated based on the vector_front.
    updated_rot_mat = m.change_mat_bbox(rot_mat, vector_front)
    bbox.update_rotation_matrix(updated_rot_mat)

    # Calculate the angle between the x-axis of the bbox and the x-axis in inertial coordinates.
    teta_p = m.matrix_to_angle(updated_rot_mat)
    teta_p = m.normalize_angle(teta_p)

    # Calculate the angle between the vector from the center to the next checkpoint and the x-axis in inertial coordinates.
    teta_t = m.points_to_angle(center, next_checkpoint)
    teta_t = m.normalize_angle(teta_t)

    # Calculate the orientation error angle that the robot needs to rotate to face the first checkpoint.
    teta_e = teta_t - teta_p

    position,calc_angle=next_point_first_time(bbox,teta_e,delta_t,v_max,beta)
    return (position, calc_angle)

def other_times(bbox, teta_e, checkpoints, delta_t, v_max, beta):
    """
    Calculates the next position and orientation angle for the robot in subsequent iterations.

    Parameters:
    bbox (o3d.geometry.OrientedBoundingBox()): Representation of an object in 3D space.
    teta_e (float): The orientation error angle (in degrees) from the previous iteration.
    checkpoints (np.array): Equally spaced checkpoints of the path.
    delta_t (float): The time step.
    v_max (float): The maximum velocity that the robot can have.
    beta (float): Constant used to calculate the adjustment angle for each iteration.

    Returns:
    position (np.array): (x, y, z) position of the robot center in the next iteration.
    calc_angle (float): The angle (in degrees) that the robot is rotating in the next iteration.
    """

    # Store the previous orientation error angle for reference.
    old_teta_e = teta_e

    # Calculate the new orientation error angle based on the current bounding box and checkpoints.
    teta_e = get_teta_e(bbox, checkpoints)

    # Calculate the next position and orientation angle for the robot.
    position, calc_angle = next_point_other_times(bbox, old_teta_e, teta_e, delta_t, v_max, beta)

    return position, calc_angle

"""_____Funções relativamente ao desvio de objectos___"""
def is_box_in_path(bbox, path):
    """
    Checks if box is in path.

    Parameters:
    bbox (o3d.geometry.OrientedBoundingBox()): Representation of an object in the space.
    path (np array): Representation of the path in a np array format

    Returns:
    bool: True if the any of the path (x,y) cordinates are the same as the box, False otherwise.
    """
    center = bbox.get_center()
    #print("Center", center)
    extent = bbox.get_extent()
    #print("extent",extent)
    #print ("path",path)
    #print("extent",extent)

    # Calculate the bounding box coordinates
    bbox_x_min = center[0] - extent[0]
    bbox_x_max = center[0] + extent[0]
    bbox_y_min = center[1] - extent[1]
    bbox_y_max = center[1] + extent[1]

    # Check if any coordinate of the bounding box is inside the path
    for point in path:
        x, y, _ = point
        if (bbox_x_min <= x <= bbox_x_max and bbox_y_min <= y <= bbox_y_max):
            #print("ENTROU NO LOOP",(x,y))
            
            return True

    return False

def find_point_index(target_point, point_array):
    """
    Find the index of the nearest point in the array to the target point.

    Parameters:
    target_point (tuple or list): The target point (x, y).
    point_array (np array): Numpy array containing points (each row is a point, columns are x and y).

    Returns:
    nearest_index (int): Index of the nearest point in the array.
    """
    target_point = np.array(target_point)
    point_array = np.array(point_array)

    # Calculate the Euclidean distance between the target point and all points in the array
    distances = np.linalg.norm(point_array - target_point, axis=1)

    # Find the index of the point with the minimum distance
    nearest_index = np.argmin(distances)

    return nearest_index

def find_deviation_point(object, close_check, robot, margin=0.5):
    """
    Finds a point to deviate from the current trajectory to avoid collisions with an object.

    Parameters:
    object (o3d.geometry.OrientedBoundingBox()): Representation of an object in 3D space.
    close_check (np.array): Closest checkpoint pair in the path.
    robot (o3d.geometry.OrientedBoundingBox()): Representation of the robot in 3D space.
    margin (float, optional): Safety margin to adjust the deviation point. Default is 0.5.

    Returns:
    deviation_point (np.array): (x, y, z) coordinates of the deviation point.
    """

    # Get the extent of the robot for safety margin considerations.
    robot_ext = robot.get_extent()

    # Calculate the vector along the path between the two closest checkpoints.
    vector_path = m.create_vector(close_check[0], close_check[1])

    # Calculate the perpendicular vector to the path.
    perp_vector = (vector_path[1], -vector_path[0])

    # Calculate the deviation point coordinates based on the object, robot, and safety margin.
    point_x = object.get_center()[0] + perp_vector[0] * (robot_ext[0] + object.get_extent()[0] + margin)
    point_y = object.get_center()[1] + perp_vector[1] * (robot_ext[1] + object.get_extent()[1] + margin)

    # Create a 3D point for the deviation.
    deviation_point = np.array((point_x, point_y, 0))

    return deviation_point

def find_closest_checkpoints(position, checkpoints, tresh=0.8, index_offset=0):
    """
    Finds the two closest checkpoints to a given position while considering a distance threshold.

    Parameters:
    position (np.array): Current position of the robot (x, y, z).
    checkpoints (np.array): Array of checkpoints in the path.
    tresh (float, optional): Distance threshold for considering checkpoints. Default is 0.8.
    index_offset (int, optional): Offset for adjusting the indices. Default is 0.

    Returns:
    close_checkpoints (tuple): Tuple containing the two closest checkpoints as np.arrays.
    """

    # Calculate distances from the position to all checkpoints in the path.
    distances = np.linalg.norm(checkpoints[:, :2] - np.array(position[:2]), axis=1)

    # Enumerate to retain both index and distance in the iteration.
    enumerated_distances = list(enumerate(distances))
    # Sort by distance.
    sorted_distances = sorted(enumerated_distances, key=lambda x: x[1])
    # Take the indices of the two closest checkpoints.
    indices_of_closest = [index for index, _ in sorted_distances[:2]]

    # Calculate indices for the previous and next checkpoints, considering the offset.
    previous_index = min(indices_of_closest) - index_offset
    next_index = max(indices_of_closest) + index_offset

    print("previous check", previous_index)
    print("next_check", next_index)

    # Check if the distances of the previous and next are greater than the threshold.
    if distances[previous_index] >= tresh and distances[next_index] >= tresh:
        return checkpoints[previous_index], checkpoints[next_index]
    else:
        # Recursively call the function with an increased index_offset.
        return find_closest_checkpoints(position, checkpoints, tresh, index_offset + 1)

def update_path(closest_check, outside_point, original_path, spacing):
    """
    Updates the path by inserting two straight lines between the closest checkpoints and an outside point.

    Parameters:
    closest_check (tuple): Tuple containing the two closest checkpoints as np.arrays.
    outside_point (np.array): Point outside the path where the straight lines connect.
    original_path (np.array): Original path as a sequence of waypoints.
    spacing (float): Spacing between points on the straight lines.

    Returns:
    new_path (np.array): Updated path after inserting straight lines.
    """

    # Create two straight lines connecting the closest checkpoints to the outside point.
    line1 = m.create_straight_line(closest_check[0][:2], outside_point[:2], spacing)
    line2 = m.create_straight_line(outside_point[:2], closest_check[1][:2], spacing)

    # Find indices of the closest checkpoints in the original path.
    prev_index = np.where(np.all(original_path[:, :2] == np.array(closest_check[0])[:2], axis=1))[0][0]
    next_index = np.where(np.all(original_path[:, :2] == np.array(closest_check[1])[:2], axis=1))[0][0]

    # Extract segments of the original path before and after the closest checkpoints.
    previous_path = original_path[:prev_index]
    next_path = original_path[next_index + 1:]

    # Merge the segments and the straight lines to create the updated path.
    new_path = m.merge_arrays(previous_path, line1, line2, next_path)
    
    return new_path

def reupdate_path(closest_check, original_path, spacing):
    """
    Re-updates the path by replacing the segment between the closest checkpoints with a straight line.

    Parameters:
    closest_check (tuple): Tuple containing the two closest checkpoints as np.arrays.
    original_path (np.array): Original path as a sequence of waypoints.
    spacing (float): Spacing between points on the straight line.

    Returns:
    new_path (np.array): Updated path after replacing the segment with a straight line.
    """

    # Find indices of the closest checkpoints in the original path.
    prev_index = find_point_index(closest_check[0], original_path)
    next_index = find_point_index(closest_check[1], original_path)

    # Extract segments of the original path before and after the closest checkpoints.
    previous_path = original_path[:prev_index]
    next_path = original_path[next_index:]

    # Create a straight line connecting the closest checkpoints.
    line = m.create_straight_line(closest_check[0][:2], closest_check[1][:2], spacing)

    # Merge the segments and the straight line to create the re-updated path.
    new_path = m.merge_arrays(previous_path, line, next_path)
    
    return new_path

def create_3d_bounding_box(center, extent, rotation_matrix, scale_factor=1.2):
    """
    Create the vertices of a scaled 3D bounding box.

    Parameters:
    center (np array): Center of the bounding box.
    extent (np array): Extent of the bounding box along x, y, and z axes.
    rotation_matrix (np array): 3x3 rotation matrix defining the orientation of the bounding box.
    scale_factor (float, optional): Scale factor to adjust the size of the bounding box. Defaults to 1.2.

    Returns:
    vertices (np array): Numpy array containing the vertices of the scaled bounding box.
    """
    # Define half extents for convenience
    half_extents = extent / 2.0

    # Define local coordinates of the bounding box
    local_vertices = np.array([
        [-1, -1, -1],
        [-1, -1,  1],
        [-1,  1, -1],
        [-1,  1,  1],
        [ 1, -1, -1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [ 1,  1,  1]
    ])

    # Apply rotation matrix to local coordinates
    rotated_vertices = np.dot(local_vertices, rotation_matrix.T)

    # Scale by the half extents, adjust size with scale factor, and translate to the center
    scaled_and_translated_vertices = (rotated_vertices * half_extents * scale_factor) + center

    return scaled_and_translated_vertices
