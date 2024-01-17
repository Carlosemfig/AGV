import open3d as o3d
import MAIN_BG_subtraction as b
import pickle
import numpy as np
import MAIN as m
import Class_object as c
from sklearn.cluster import DBSCAN
import time

eps = 0.32
min_samples = 20 
distance=0.2

tracker=c.EuclideanDistTracker3D()
# Load voxel grid from the saved file
bg = o3d.io.read_voxel_grid("saved_voxel_grid.ply")


first_path= r'C:\Users\hvendas\Desktop\agv-snr\data_dict_track_1.pkl'
# Load the data from the Pickle file
with open(first_path, 'rb') as file:
    loaded_data_dict_obj = pickle.load(file)

num_arrays_to_merge_obj =50
i=0
merged_array_obj = []
for key, value in loaded_data_dict_obj.items():
    if i < num_arrays_to_merge_obj:
        merged_array_obj.append(value)
        #print("merged array",merged_array)
        i=i+1

obj_1 = np.concatenate(merged_array_obj, axis=0)


result_1 = b.points_outside_all_voxels(obj_1, bg)



# Perform clustering
cluster_labels, num_clusters = m.perform_clustering(result_1, eps, min_samples)



start_time = time.time()
# Call centroid_and_box function to get visualizations of centroids and bounding boxes
all_visualizations, bbox_info = m.centroid_and_box(result_1, cluster_labels, num_clusters)
end_time = time.time()
execution_time = end_time - start_time
print("TIME",execution_time)
tracker.update(bbox_info,distance)
tracker.print_stored_objects()


# Convert the result array to a point cloud
result_point_cloud = o3d.geometry.PointCloud()
result_point_cloud.points = o3d.utility.Vector3dVector(result_1)

# Visualize the result along with centroids and bounding boxes
o3d.visualization.draw_geometries([result_point_cloud] + all_visualizations)










second_path= r'C:\Users\hvendas\Desktop\agv-snr\data_dict_track_2.pkl'

with open(second_path, 'rb') as file:
    loaded_data_dict_obj = pickle.load(file)

num_arrays_to_merge_obj =50
i=0
merged_array_obj = []
for key, value in loaded_data_dict_obj.items():
    if i < num_arrays_to_merge_obj:
        merged_array_obj.append(value)
        #print("merged array",merged_array)
        i=i+1

obj_2 = np.concatenate(merged_array_obj, axis=0)
result_2 = b.points_outside_all_voxels(obj_2, bg)

# Perform clustering
cluster_labels, num_clusters = m.perform_clustering(result_2, eps, min_samples)

# Call centroid_and_box function to get visualizations of centroids and bounding boxes
all_visualizations, bbox_info = m.centroid_and_box(result_2, cluster_labels, num_clusters)

tracker.update(bbox_info,distance)
tracker.print_stored_objects()

# Convert the result array to a point cloud
result_point_cloud = o3d.geometry.PointCloud()
result_point_cloud.points = o3d.utility.Vector3dVector(result_2)

# Visualize the result along with centroids and bounding boxes
o3d.visualization.draw_geometries([result_point_cloud] + all_visualizations)


"""

pc_result_1=m.array_to_pc(result_1)
m.visualize(pc_result_1)
pc_result_2=m.array_to_pc(result_2)
m.visualize(pc_result_2)
"""