import numpy as np


def ajust_angle(beta, teta_e, delta_t):
    ang_vel=beta*teta_e
    teta_ajust=ang_vel*delta_t
    return(teta_ajust)
                             
# Define the first rotation matrix
rotation_matrix1 = np.array([[0.5, 0, -0.8660254],
                             [-0.8660254, 0, -0.5],
                             [0, 1, 0]])

# Specify the 180-degree rotation matrix around the z-axis
rotation_180_degrees = np.array([[-1, 0, 0],
                                 [0, -1, 0],
                                 [0, 0, 1]])

# Obtain the new rotation matrix by applying the 180-degree rotation
new_rotation_matrix = np.dot( rotation_180_degrees,rotation_matrix1)

pointing_direction=np.dot(new_rotation_matrix, np.array([1,0,0]))


teta_e=10





print("pointing_direction",pointing_direction)


print("Original rotation matrix:")
print(rotation_matrix1)

print("\n180-degree rotation matrix around the z-axis:")
print(rotation_180_degrees)

print("\nNew rotation matrix:")
print(new_rotation_matrix)



def find_closest_checkpoint(position, checkpoints):
    """
    Finds the next checkpoint in the given list of checkpoints.

    Parameters:
    position (np.array): The (x, y, z) current position of the robot center.
    checkpoints (np.array): An array with all the checkpoints.

    Returns:
    next_checkpoint (np array): The (x, y, z) position of the next checkpoint.
    """

    closest_point = None
    min_distance = float('inf')

    for point in checkpoints:
        distance = np.linalg.norm(np.array(position[:2]) - np.array(point[:2]))

        if distance < min_distance:
            min_distance = distance
            closest_point = point

    # Find the index of the closest point
    index_closest_point = np.argmin(np.linalg.norm(checkpoints[:, :2] - np.array(position[:2]), axis=1))

    # Ensure that the index is not the last index to avoid index out of range
    if index_closest_point < len(checkpoints) - 1:
        # Return the next checkpoint in the list
        next_checkpoint = checkpoints[index_closest_point + 1]
    else:
        # Return the last checkpoint if the current one is the last in the list
        next_checkpoint = checkpoints[index_closest_point]

    return next_checkpoint

# Example usage:
checkpoints = np.array([[1., 0., 0.], [1., 1., 0.], [1., 2., 0.], [1., 3., 0.], [1., 4., 0.], [1., 5., 0.], [1., 6., 0.],
                        [2., 6., 0.], [3., 6., 0.], [4., 6., 0.], [4., 5., 0.], [4., 4., 0.], [4., 3., 0.], [4., 2., 0.],
                        [4., 1., 0.], [4., 0.1, 0.]])

position = np.array([1., 0.3, 0.])

next_checkpoint = find_closest_checkpoint(position, checkpoints)
print("next checkpoint",next_checkpoint)