import numpy as np
def erase_previous_path(path, point):
    """
    Given the path and a point, the path is erased up to that point and the rest of the path is returned.~

    Parameters: 
        path(np.array)= total path that is meant to be trimmed
        point(x,y,z) or (x,y)= the point in where the path should be trimmed
        the point format should be the same as the points in the path.

    Returns: 
        trimmed_path(np.array): the path from the point forward

    """
    # Convert the path and checkpoint to NumPy arrays for efficient calculations
    path_array = np.array(path)
    point_array = np.array(point)

    # Calculate the distances between the checkpoint and all points in the path
    distances = np.linalg.norm(path_array - point_array, axis=1)

    # Find the index of the closest point
    closest_index = np.argmin(distances)

    # Return the points with indexes equal to or greater than the closest index
    trimmed_path = path[closest_index:]
    
    return trimmed_path

# Example usage:
path = [(1, 2,1),(4,5,1) ,(3, 4,1), (5, 6,1), (7, 8,1)]
checkpoint = (4, 5,1)

result = erase_previous_path(path, checkpoint)
print("Trimmed path:", result)