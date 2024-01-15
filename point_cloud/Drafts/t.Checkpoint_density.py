import MAIN as m
import numpy as np

line1=m.create_straight_line((0,1),(0,3),0.1)
line2=m.create_straight_line((0,3),(1,3),0.1)
path=m.merge_arrays(line1,line2)



#print (path)

checkpoints_original=m.create_checkpoints(path,4)
print("original",checkpoints_original)
#checkpoints_new=create_checkpoints_new(path,4)
#print("novo",checkpoints_new)


def index_of_curves_in_path(path, threshold=1e-10):
    vectors = np.diff(path, axis=0)
    data=np.diff(vectors,axis=0)
    non_zero_indices = []

    for i, vector in enumerate(data):
        # Use numpy to check if the vector is different from the zero vector
        if np.linalg.norm(vector) > threshold:
            non_zero_indices.append(i+2)

    return non_zero_indices


import numpy as np

def create_checkpoints_with_variable_spacing(path, spacing_away=4, spacing_near=2, z_offset=0.0, threshold=0.5):
    """
    Creates checkpoints along the given path with variable spacing near provided indexes.

    Parameters:
    path (np array): The path represented as a numpy array.
    indexes (list): List of indexes where the spacing should be different.
    spacing_away (int, optional): Spacing away from the indexes. Defaults to 1.0.
    spacing_near (int, optional): Spacing near the indexes. Defaults to 0.1.
    z_offset (float, optional): The offset in the z-direction. Defaults to 0.0.
    threshold (float, optional): Threshold to determine what is considered "near" the indexes. Defaults to 1e-10.

    Returns:
    checkpoints (np array): Numpy array representing checkpoints along the path.
    """
    # Ensure that the path is a numpy array
    
    path = np.array(path)
    indexes=index_of_curves_in_path(path)
    checkpoints = []
    
    # Iterate over the path
    for i in range(len(path)):
        # Check if the current index is near any provided indexes
        near_index = any(np.linalg.norm(path[i] - path[idx]) < threshold for idx in indexes)

        # Choose the appropriate spacing based on whether the current index is near or away from provided indexes
        current_spacing = spacing_near if near_index else spacing_away

        # Add checkpoint with the chosen spacing
        if i == 0 or i == len(path) - 1 or i % int(current_spacing) == 0:
            checkpoint = path[i].copy()
            checkpoint[2] += z_offset
            checkpoints.append(checkpoint)

    return np.array(checkpoints)


indexes_of_interest = index_of_curves_in_path(path)
result_checkpoints = create_checkpoints_with_variable_spacing(path)

print("Resulting checkpoints:")
print(result_checkpoints)






