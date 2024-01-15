import pickle
import MAIN as m
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
import open3d as o3d
import matplotlib.pyplot as plt

# Specify the path to the Pickle file
file_path = r'C:\Users\hvendas\Desktop\agv-snr\data_dict_new.pkl'


# Load the data from the Pickle file
with open(file_path, 'rb') as file:
    loaded_data_dict = pickle.load(file)

print("Comprimento",len(loaded_data_dict))
"""
first_timestamp = list(loaded_data_dict.keys())[0]
first_array = loaded_data_dict[first_timestamp]
pc=m.array_to_pc(first_array)



"""


# Merge all arrays in the dictionary
#merged_array = np.concatenate(list(loaded_data_dict.values()), axis=0)




num_arrays_to_merge =1
i=0
merged_array = []

for key, value in loaded_data_dict.items():
    if i < num_arrays_to_merge:
        merged_array.append(value)
        #print("merged array",merged_array)
        i=i+1

merged_array = np.concatenate(merged_array, axis=0)


min_values = np.min(merged_array, axis=0)
max_values = np.max(merged_array, axis=0)



print("Minimum values for each column:", min_values)
print("Maximum values for each column:", max_values)

midpoint_x = (min_values[0] + max_values[0]) / 2
midpoint_y = (min_values[1] + max_values[1]) / 2
midpoint_z = (min_values[2] + max_values[2]) / 2

# Create a tuple or array representing the midpoint
midpoint = (midpoint_x, midpoint_y, midpoint_z)

print("Midpoint:", midpoint)

"""# Eliminate repeated points
merged_array, unique_indices = np.unique(merged_array, axis=0, return_index=True)
np.set_printoptions(threshold=np.inf)
with open("merged_array_output_new.txt", "w") as f:
    print(merged_array, file=f)"""

pc=m.array_to_pc(merged_array)
m.visualize(pc)


"""

# Step 1: Initialize DBSCAN

dbscan = DBSCAN(eps=0.3, min_samples=30)

# Step 2: Fit and predict clusters
labels = dbscan.fit_predict(merged_array)

np.set_printoptions(threshold=np.inf)
# Save the labels to a file
with open("labels_output.txt", "w") as f:
    print(labels, file=f)

# Step 3: Filter out points labeled as outliers (-1)
# Step 3: Filter out points labeled as outliers (-1)
filtered_array = merged_array[labels != -1]
filtered_labels = labels[labels != -1]

#filtered_array=merged_array
#filtered_labels=labels

# Convert the filtered array to Open3D point cloud
filtered_pc = o3d.geometry.PointCloud()
filtered_pc.points = o3d.utility.Vector3dVector(filtered_array)

# Paint each point in the point cloud with a unique color based on the cluster labels
unique_labels = np.unique(filtered_labels)
colors = plt.get_cmap("tab10")(unique_labels % 10)

# Map original labels to colors
color_mapping = {label: color for label, color in zip(unique_labels, colors)}
mapped_colors = np.array([color_mapping[label] for label in filtered_labels])

# Assign the colors to the point cloud
filtered_pc.colors = o3d.utility.Vector3dVector(mapped_colors[:, :3])

# Visualize the point cloud with colored clusters
o3d.visualization.draw_geometries([filtered_pc], point_show_normal=False)"""