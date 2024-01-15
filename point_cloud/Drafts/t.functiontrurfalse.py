import open3d as o3d
import numpy as np
import MAIN as m

# Define the center and extent as NumPy arrays
center = np.array([1.0, 1.0, 0.25])
extent = np.array([0.6, 0.6, 0.5])

# Create the OrientedBoundingBox
object_bbox = o3d.geometry.OrientedBoundingBox(center, np.identity(3), extent)


# Define the center and extent as NumPy arrays
center_2 = np.array([-1.0, -1.0, 0.25])
extent_2 = np.array([0.6, 0.6, 0.5])



# Create the OrientedBoundingBox
object_bbox_2 = o3d.geometry.OrientedBoundingBox(center_2, np.identity(3), extent_2)




print("este é o centro", object_bbox.center)
print("este é a dimensão", object_bbox.extent)

# Define the starting and ending coordinates of the three straight lines
line1_start = (1.0, 0.0)
line1_end = (1.0, 6.0)

line2_start = (1.0, 6.0)
line2_end = (4.0, 6.0)

line3_start = (4.0, 6.0)
line3_end = (4.0, 0.0)

# Generate the three straight lines
spacing = 0.1
x_path1, y_path1, z_path1 = m.create_straight_line(line1_start[0], line1_start[1], line1_end[0], line1_end[1], spacing)
x_path2, y_path2, z_path2 = m.create_straight_line(line2_start[0], line2_start[1], line2_end[0], line2_end[1], spacing)
x_path3, y_path3, z_path3 = m.create_straight_line(line3_start[0], line3_start[1], line3_end[0], line3_end[1], spacing)

global_path= np.vstack((np.column_stack((x_path1.flatten(), y_path1.flatten(), z_path1.flatten())), np.column_stack((x_path2.flatten(), y_path2.flatten(), z_path2.flatten()))))
global_path= np.vstack((global_path,np.column_stack((x_path3.flatten(), y_path3.flatten(), z_path3.flatten()))))




def is_box_in_path(bbox, path):
    """
    Checks if box is in path.

    Input:
    bbox: is a o3d.geometry.OrientedBoundingBox() object
    path: ia a numpy array with 3 columns

    Returns:
    True: if the coordinates of the path are the same as the bbox
    False: if not
    """
    center = bbox.center
    extent = bbox.extent

    # Calculate the bounding box coordinates
    bbox_x_min = center[0] - extent[0]
    bbox_x_max = center[0] + extent[0]
    bbox_y_min = center[1] - extent[1]
    bbox_y_max = center[1] + extent[1]

    # Check if any coordinate of the bounding box is inside the path
    for point in path:
        x, y, _ = point
        if (bbox_x_min <= x <= bbox_x_max and bbox_y_min <= y <= bbox_y_max):
            print(x,y)
            return True

    return False



# Check if the bounding box intersects the path
is_intersecting = is_box_in_path(object_bbox_2, global_path)

is_box_in_path()
print(is_intersecting)
if is_intersecting:
    print("The bounding box intersects the path.")
else:
    print("The bounding box does not intersect the path.")


"""def is_object_extent_on_path(object_bbox, global_path):
    # Get the extent of the bounding box
    extent = object_bbox.extent

    # Check if the extent is within the boundaries of the path
    min_x, max_x = global_path[:, 0].min(), global_path[:, 0].max()
    min_y, max_y = global_path[:, 1].min(), global_path[:, 1].max()
    min_z, max_z = global_path[:, 2].min(), global_path[:, 2].max()

    is_extent_on_path = (
        min_x <= extent[0] <= max_x and
        min_y <= extent[1] <= max_y and
        min_z <= extent[2] <= max_z
    )

    return is_extent_on_path

# Check if the extent of the object_bbox is on the global_path
is_extent_on_path = is_object_extent_on_path(object_bbox, global_path)

if is_extent_on_path:
    print("True")
else:
    print("False")"""