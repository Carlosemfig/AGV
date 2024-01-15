import open3d as o3d
from sklearn.cluster import DBSCAN
import numpy as np
import math

def load_pc(file_path):
    # Input is the file path
    # Output is the point cloud in a array shape
    pc = o3d.io.read_point_cloud(file_path)
    return pc

def save_pc(point_cloud, file_path):
    # Input is the point cloud to be saved and the path to save it
    o3d.io.write_point_cloud(file_path, point_cloud)

def subtract_array(bg_points, object_points):
    # Input is the arrays for both the bg and the map with objects
    # Output is the resulting point cloud in array format
    result_points = np.array([point for point in object_points if not np.any(np.all(point == bg_points, axis=1))])
    return result_points

def pc_to_array(point_cloud):
    # Input is a point cloud
    # Output is the point cloud in array format
    array=np.asarray(point_cloud.points)
    return array

def array_to_pc(array):
    # Input is a point cloud in array format
    # Output is the point cloud
    result_cloud = o3d.geometry.PointCloud()
    result_cloud.points = o3d.utility.Vector3dVector(array)
    return result_cloud
 
def perform_clustering(points, eps, min_samples):
    # Input is a array form point cloud, 
    # Input is eps(maximum distance between points in the same cluster)
    # Input is the minimum number os points to form a cluster
    # Output is a array form of the clusters and the number of clusters
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    cluster_labels = clustering.labels_
    num_clusters= len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    return cluster_labels, num_clusters

def create_checkpoints(path, checkpoint_spacing):
    x_checkpoints = path[0][::checkpoint_spacing]
    y_checkpoints = path[1][::checkpoint_spacing]
    z_checkpoints = path[2][::checkpoint_spacing]
    return np.vstack((x_checkpoints, y_checkpoints, z_checkpoints)).T
#def find_closest_checkpoint(position, path):
 #   closest_point = None
 #   min_distance = float('inf')

#    for point in path:
#        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(position, point)))
#        if distance < min_distance:
#            min_distance = distance
#            closest_point = point
#    return closest_point

def find_closest_checkpoint(position, path):
    closest_point = None
    min_distance = float('inf')

    for point in path:
        distance = np.linalg.norm(np.array(position[:2]) - np.array(point[:2]))

        if distance < min_distance:
            min_distance = distance
            closest_point = point

    # Find the index of the closest point
    index_closest_point = np.argmin(np.linalg.norm(path[:, :2] - np.array(position[:2]), axis=1))

    # Ensure that the index is not the last index to avoid index out of range
    if index_closest_point < len(path) - 1:
        # Check if the next point is on the direction to the end of the path
        vector_to_point = np.array(closest_point[:2]) - np.array(position[:2])
        vector_to_end = np.array(path[-1][:2]) - np.array(position[:2])

        dot_product = np.dot(vector_to_point, vector_to_end)

        if dot_product > 0:
            return path[index_closest_point]

        # If not, return the next point on the path
        return path[index_closest_point + 1]

    return path[index_closest_point]

def centroid_and_box(points,cluster_labels,num_clusters):
    # Input is points (the point cloud that is being analysed)
    # Input is the cluster_labels and num_clusters resulting from the perform clustering function
    # Output is the list with all the point clouds to add to the visualizer
    all=[]
    bbox={}
    
    for cluster_id in range(num_clusters):
        #print(cluster_id)
        #print(cluster_labels)
        object_points = points[cluster_labels == cluster_id]
        
        # Calculate the centroid of the object
        centroid = np.mean(object_points, axis=0)
        


        # Calculate the bounding box of the object
        min_bound = np.min(object_points, axis=0)
        max_bound = np.max(object_points, axis=0)

        # Create a colored point cloud for the object's centroid
        centroid_color = [0, 0, 1]  # Blue color
        centroid_cloud = o3d.geometry.PointCloud()
        centroid_cloud.points = o3d.utility.Vector3dVector([centroid])
        centroid_cloud.colors = o3d.utility.Vector3dVector([centroid_color])

        # Create a colored bounding box for the object
        object_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(object_points))
        
        object_bbox.color = [0, 1, 0]  # Green color
        center=object_bbox.center
        extend=object_bbox.extent

        # Add the objects to the visualizer
        all.append(centroid_cloud)
        all.append(object_bbox)
        bbox [cluster_id]= (center,extend)
    
    return all, bbox

def visualize(point_cloud):
    # The Input is a pointcloud structure
    o3d.visualization.draw_geometries([point_cloud])

def run_visualizer(point_clouds):
    # Input is a list of point clouds to be visualized
    # Runs the visualizer and opens a window
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for point_cloud in point_clouds:
        vis.add_geometry(point_cloud)

    vis.run()
    vis.destroy_window() 

def subtract_main(bg_path,obj_path,result_path="result.pcd"):
    all=[]
    # Input the path for the bg, objects and where to save the resulting file
    # Output saves the resulting file and allows to visualize it
    bg=load_pc(bg_path)
    obj=load_pc(obj_path)

    bg=bg.points
    obj=obj.points

    result=subtract_array(bg,obj)

    result=array_to_pc(result)

    save_pc(result, result_path)
    all.append(result)
    run_visualizer(all)

def object_detection_main(result_path):
    # Input is the path for the point cloud that is meant to be analysed
    # Runs the visualizer and shows the bounding boxes and centroids of the objects
    
    cloud= load_pc(result_path)
    point_cloud = pc_to_array(cloud)

    Eps=0.2
    Min_samples=10


    Labels, Number=perform_clustering(point_cloud,Eps,Min_samples)

    all, bbox= centroid_and_box(point_cloud,Labels,Number)

    all.append(cloud)
    run_visualizer(all)

def get_bbox(map):
    """
    Function that retrives the bbox.

    Parameters:
    map (np array): Is the map where we want to retrieve the bbox.

    Returns:
    bbox (dict): The keys are the box id and for each key (center,extent).
    """
    
    Eps=0.2
    Min_samples=10


    Labels, Number=perform_clustering(map,Eps,Min_samples)

    all,bbox = centroid_and_box(map,Labels,Number)

    return bbox

def merge_arrays(*arrays):
    """
    Merges multiple arrays into a single array.

    Parameters:
    *arrays (np array): Variable number of arrays to be merged.

    Returns:
    merged_array (np array): The merged array.
    """
    merged_array = np.vstack(arrays)
    return merged_array

def random_coordinate_from_array(array):
    """
    Function that gets a random pair of coordinates from a given array. 

    Parameters:
    array (np array): Can be a path or a plane for example.

    Returns:
    x,y (tuple): Random coordinates belonging to the given array.
    """

    # Get the number of rows in the global path
    num_rows = array.shape[0]

    # Generate a random index within the range of available rows
    random_index = np.random.randint(0, num_rows)

    # Get the x and y coordinates from the randomly selected row
    x, y, _ = array[random_index]
    #print(random_index)

    return x, y

def create_straight_line(line_start, line_end, spacing):
    """
    Creates a np_array of a strait line. 

    Parameters:
    line_start (tuple): Is the (x,y) cordinates where the line starts.
    line_end (tuple): Is the (x,y) cordinates where the line ends.
    spacing (float): Is the space between the represented points of the line.

    Returns:
    path (np array): A np_array representation of a line. 
    """
    x1=line_start[0]
    y1=line_start[1]

    x2=line_end[0]
    y2=line_end[1]
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    # Calculate the number of points needed based on the spacing
    num_points = int(distance / spacing)
    
    # Generate the path points
    x_path = np.linspace(x1, x2, num_points,endpoint=False)
    
    y_path = np.linspace(y1, y2, num_points,endpoint=False)
    z_path = np.zeros(num_points)  # Set the z-coordinates to be at floor level
    
    path = np.column_stack((x_path, y_path, z_path))
    return path

def create_cubic_object(center, size_x,size_y, spacing):
    """
    Creates a np_array of a cubic object. 

    Parameters:
    center (tuple): Is the (x,y) cordinates for the center of the cube.
    size_x (float): Is the value of the width (x-direction) of the cube.
    size_y (float): Is the value of the lenght (y-direction), and height (z_direction) of the cube.
    spacing (float): Is the space between the represented points of the cube.

    Returns:
    object (np array): A np_array representation of a cube. 
    """

    # Calculate the dimensions of the cubic object
    h_width= size_x/2
    h_lenght=size_y/2
    #h_lenght=length/2
    #print(round(center[0],1),round(center[1], 1))


    width_range = np.arange(round(center[0],1) - h_width, round(center[0],1) + h_width +0.0000000001, spacing)
    length_range = np.arange(round(center[1], 1) - h_lenght, round(center[1], 1) + h_lenght +0.0000000001, spacing)
    height_range = np.arange(0, size_y, spacing)

    #print("width and lenght",width_range,length_range)
    # Create a grid of points to represent the cubic object
    x, y, z = np.meshgrid(width_range, length_range, height_range, indexing='ij')

    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    object=np.column_stack((x, y, z))

    return object

def update_path(current_position, next_checkpoint, original_path, spacing):
    """
    Updates the path based on the current position, next checkpoint, and original path.

    Parameters:
    current_position (tuple): Current position (x, y, z) of the robot.
    next_checkpoint (tuple): Next checkpoint (x, y, z) on the original path.
    original_path (np array): Original path as a numpy array.
    spacing (float): Spacing between points in the path.

    Returns:
    updated_path (np array): Updated path after the current position and next checkpoint.
    """
    # Create a straight line between the current position and the next checkpoint
    straight_line = create_straight_line(current_position[:2], next_checkpoint[:2], spacing)

    # Find the index of the next checkpoint in the original path
    checkpoint_index = np.where(np.all(original_path[:, :2] == np.array(next_checkpoint)[:2], axis=1))[0][0]

    # Slice the original path from the current checkpoint to the end
    remaining_path = original_path[checkpoint_index:]

    # Combine the straight line and the remaining path
    updated_path = np.vstack((straight_line, remaining_path))

    return updated_path


def constant_x(length, height, plane, spacing):
    """
    Creates a np_array of a x-plane. 

    Parameters:
    lenght (tuple): (a,b), where a is the first point and b is the last in y-direction.
    height (tuple): (a,b), where a is the first point and b is the last in z-direction.
    plane (float): Is the value of x where the plane is to be placed.
    spacing (float): Is the space between the represented points of the plane.

    Returns:
    object (np array): A np_array representation of a plane. 
    """
    # Create a grid of points to represent the floor
    y = np.arange(length[0], length[1], spacing)
    z= np.arange(height[0], height[1], spacing)
    y, z = np.meshgrid(y, z)
    x = np.zeros_like(y)  # Set the Z coordinate to zero for the floor
    if plane != 0:
        x[x == 0] = plane

    
    # Flatten the grid to create a point cloud
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    plane=np.column_stack((x, y, z))

    return plane

def constant_y(width, height, plane, spacing):
    """
    Creates a np_array of a y-plane. 

    Parameters:
    width (tuple): (a,b), where a is the first point and b is the last in x-direction.
    height (tuple): (a,b), where a is the first point and b is the last in z-direction.
    plane (float): Is the value of y where the plane is to be placed.
    spacing (float): Is the space between the represented points of the plane.

    Returns:
    object (np array): A np_array representation of a plane. 
    """
    # Create a grid of points to represent the floor
    x = np.arange(width[0], width[1], spacing)
    z= np.arange(height[0], height[1], spacing)
    x, z = np.meshgrid(x, z)
    y = np.zeros_like(x)  # Set the Z coordinate to zero for the floor
    if plane != 0:
        y[y == 0] = plane

    
    # Flatten the grid to create a point cloud
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    plane=np.column_stack((x, y, z))

    return plane

def constant_z(width, length, plane, spacing):
    """
    Creates a np_array of a z-plane. 

    Parameters:
    width (tuple): (a,b), where a is the first point and b is the last in x-direction.
    lenght (tuple): (a,b), where a is the first point and b is the last in y-direction.
    plane (float): Is the value of z where the plane is to be placed.
    spacing (float): Is the space between the represented points of the plane.

    Returns:
    object (np array): A np_array representation of a plane. 
    """
    # Create a grid of points to represent the floor
    x = np.arange(width[0], width[1], spacing)
    y = np.arange(length[0], length[1], spacing)
    x, y = np.meshgrid(x, y)
    z = np.zeros_like(x)  # Set the Z coordinate to zero for the floor
    if plane != 0:
        z[z == 0] = plane

    
    # Flatten the grid to create a point cloud
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    plane=np.column_stack((x, y, z))

    return plane

def is_box_in_path(bbox, path):
    """
    Checks if box is in path.

    Parameters:
    bbox (o3d.geometry.OrientedBoundingBox()): Representation of an object in the space.
    path (np array): Representation of the path in a np array format

    Returns:
    bool: True if the any of the path (x,y) cordinates are the same as the box, False otherwise.
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
            
            return True

    return False

def is_center_in_path(center, path):
    """
    Checks if the center of the box coordinates are in path.

    Parameters:
    center (tuple): (x,y) are the center cordinates of the box.
    path (np array): Representation of the path in a np array format

    Returns:
    bool: True if the any of the path (x,y) cordinates are the same as the box, False otherwise.
    """
    x_box,y_box=center

    # Check if any coordinate of the bounding box is inside the path
    for point in path:
        x, y, _ = point
        if (x==x_box and y==y_box):
            return True

    return False

class EuclideanDistTracker3D:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

    def update(self, objects_dict):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for index, (center, dimensions) in objects_dict.items():
            cx, cy, cz = center
            # For simplicity, use the average of dimensions for radius
            radius = sum(dimensions) / 3.0

            # Find out if that object was detected already
            same_object_detected = False
            for obj_id, pt in self.center_points.items():
                #dist = math.sqrt((cx - pt[0])**2 + (cy - pt[1])**2 + (cz - pt[2])**2)
                dist = math.sqrt((cx - pt[0])**2 + (cy - pt[1])**2)
                print(dist)
                if dist < 0.2:  # Adjust the threshold as needed
                    self.center_points[obj_id] = (cx, cy, cz)
                    objects_bbs_ids.append([cx, cy, cz, radius, obj_id])
                    same_object_detected = True
                    break

            # New object is detected; assign the ID to that object
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy, cz)
                objects_bbs_ids.append([cx, cy, cz, radius, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDs not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids



class EuclideanDistTracker3D_new:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

    def update(self, objects_dict):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for index, (center, dimensions) in objects_dict.items():
            cx, cy, cz = center
            dx, dy, dz = dimensions


            cx=round(cx,1)
            cy=round(cy,1)
            cz=round(cz,1)

            dx=round(dx,1)
            dy=round(dy,1)
            dz=round(dz,1)
            # For simplicity, use the average of dimensions for radius
            radius = sum(dimensions) / 3.0

            # Find out if that object was detected already
            same_object_detected = False
            for obj_id, (pt,dim,trajectory) in self.center_points.items():
                #dist = math.sqrt((cx - pt[0])**2 + (cy - pt[1])**2 + (cz - pt[2])**2)
                dist = math.sqrt((cx - pt[0])**2 + (cy - pt[1])**2)
                print(dist)
                if dist < 0.2:  # Adjust the threshold as needed
                    self.center_points[obj_id] = ((cx, cy, cz), (dx, dy, dz),trajectory + [(cx, cy, cz)])
                    objects_bbs_ids.append([cx, cy, cz,  dx, dy, dz, obj_id])
                    same_object_detected = True
                    break

            # New object is detected; assign the ID to that object
            if not same_object_detected:
                self.center_points[self.id_count] = ((cx, cy, cz), (dx, dy, dz),[(cx, cy, cz)])
                objects_bbs_ids.append([cx, cy, cz, dx, dy, dz, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDs not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids
    
if __name__ == "__main__":
    #subtract_main("bg.pcd","object_2.pcd")
    object_detection_main("result.pcd")
