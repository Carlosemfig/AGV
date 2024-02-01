#carregar o fundo
import pickle
import MAIN as m
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
import open3d as o3d
import matplotlib.pyplot as plt
from pygroundsegmentation import GroundPlaneFitting
import BG_MAIN as b
import statistics


voxel_size = 0.3 #decreasing this value will make the subtraction process harder
treshold=0.35 #decreasing this value more will remove less of the flor.



"""_______bg______"""

file_path = r'data_dict_background_model.pkl'


#Load background model file to ndarray
merged_array = b.loadFileToArray(file_path)
#print(len(merged_array))
merged_array= b.remove_duplicate_points(merged_array)


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
import statistics
import seaborn as sns

# Assuming 'merged_array' is your array of data
# Extract only the Z coordinates (third column)
z_values = merged_array[:, 2]

# Calculate mean, mode, and standard deviation
mean_value = np.mean(z_values)
mode_value = statistics.mode(z_values)
std_dev = np.std(z_values)

#
# Create a kernel density plot using seaborn with a smaller bandwidth
sns.kdeplot(z_values, bw=0.05, shade=True, color='blue')  # Adjust the bandwidth as needed

# Add mean, mode, and standard deviation to the plot
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
#plt.axvline(mode_value, color='green', linestyle='dashed', linewidth=2, label=f'Mode: {mode_value:.2f}')
plt.axvline(mean_value + std_dev, color='orange', linestyle='dashed', linewidth=2, label=f'Mean + Std Dev: {mean_value + std_dev:.2f}')
plt.axvline(mean_value - std_dev, color='orange', linestyle='dashed', linewidth=2, label=f'Mean - Std Dev: {mean_value - std_dev:.2f}')

# Add labels and legend
plt.xlabel('Z Values')
plt.ylabel('Probability Density')
plt.xticks(np.arange(min(z_values), max(z_values)+0.2, 0.2))
plt.legend()

# Show the plot
plt.show()

# Show the plot
plt.show()







def most_common_z_value(points):
    """
    Find the most common Z value in an array of 3D points.

    Parameters:
    - points (np.ndarray): Array of 3D points, each row representing [x, y, z]

    Returns:
    - float: Most common Z value
    """
    #z_values = points[:, 2]  # Extract the Z values from the array
    z_values = np.round(points[:, 2], decimals=1)
    print(z_values)
    most_common_z = statistics.mode(z_values)
    print("mais commum",most_common_z)
    return most_common_z

testing=most_common_z_value(merged_array)

boxes=b.get_2dboxes_from_json('BuildingsGeojson_1.json')
new_boxes=b.transform_2d_to_3d(boxes)

outside_boxes=b.points_outside_all_boxes(merged_array,new_boxes)
print("mais comum", most_common_z_value(outside_boxes))
merged_array=outside_boxes






#Ndarray surface segmentation
#segmentationArray = b.pointSegmentation(merged_array)
print(merged_array)
#Calculate Minimum and Maximum value points for Z,Y and Z dimensions
minAndMaxPointList = b.calculateMinAndMaxPoints(merged_array)
print("minimum point",minAndMaxPointList[5])
print("maximum point",minAndMaxPointList[4])
#Define the value in the Z dimension in wich all points below will be deleted
#groundThresholder = minAndMaxPointList[5] + treshold
groundThresholder = most_common_z_value(merged_array) + treshold
#Remove duplicate points
segmentationArrayNoDup = b.remove_duplicate_points(merged_array)
def remove_ground_plane(ndarray,groundThresholder):
    """
    Remove ground plane points.
 
    Parameters:
    - ndarray: Array of 3D points, each row representing [x, y, z]
    - groundThresholder: value in the Z dimension in wich all points below will be deleted
 
    Returns:
    - ground_points: Array of unique points, without the points bellow the threshold
    """
    
    # Find indices of points below the ground threshold
    ground_indices = np.where(ndarray[:, 2] > groundThresholder)[0] 
    # Extract ground points using the indices
    ground_points = ndarray[ground_indices]

    return ground_points

def remove_ground_plane_contrary(ndarray,groundThresholder):
    """
    Remove ground plane points.
 
    Parameters:
    - ndarray: Array of 3D points, each row representing [x, y, z]
    - groundThresholder: value in the Z dimension in wich all points below will be deleted
 
    Returns:
    - ground_points: Array of unique points, without the points bellow the threshold
    """
    
    # Find indices of points below the ground threshold
    ground_indices = np.where(ndarray[:, 2] <=groundThresholder)[0] 
    # Extract ground points using the indices
    ground_points = ndarray[ground_indices]

    return ground_points
#Remove ground plane points
segmentationArrayNoGroundPlane = remove_ground_plane(segmentationArrayNoDup,groundThresholder)

segmentationArrayonlyGroundPlane = remove_ground_plane_contrary(segmentationArrayNoDup,groundThresholder)





z_values = segmentationArrayonlyGroundPlane[:, 2]

# Calculate mean, mode, and standard deviation
mean_value = np.mean(z_values)
mode_value = statistics.mode(z_values)
std_dev = np.std(z_values)

#
# Create a kernel density plot using seaborn with a smaller bandwidth
sns.kdeplot(z_values, bw=0.05, shade=True, color='blue')  # Adjust the bandwidth as needed

# Add mean, mode, and standard deviation to the plot
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_value:.2f}')
#plt.axvline(mode_value, color='green', linestyle='dashed', linewidth=2, label=f'Mode: {mode_value:.2f}')
plt.axvline(mean_value + std_dev, color='orange', linestyle='dashed', linewidth=2, label=f'Mean + Std Dev: {mean_value + std_dev:.2f}')
plt.axvline(mean_value - std_dev, color='orange', linestyle='dashed', linewidth=2, label=f'Mean - Std Dev: {mean_value - std_dev:.2f}')

# Add labels and legend
plt.xlabel('Z Values')
plt.ylabel('Probability Density')
plt.xticks(np.arange(min(z_values), max(z_values)+0.05, 0.05))
plt.legend()

# Show the plot
plt.show()

# Show the plot
plt.show()












print(most_common_z_value(segmentationArrayNoGroundPlane))
##### Start Visualize point clound #####

#print(segmentationArray.size)
#print(segmentationArrayNoDup.size)

total_array=m.array_to_pc(merged_array)
m.visualize(total_array)

only_floor=m.array_to_pc(segmentationArrayonlyGroundPlane)
m.visualize(only_floor)

no_floor=m.array_to_pc(segmentationArrayNoGroundPlane)
m.visualize(no_floor)

##### End Visualize point clound #####

#Create Voxel
bg=b.voxelization(segmentationArrayNoGroundPlane,size=voxel_size)


#visualize the voxel 
o3d.visualization.draw_geometries([bg])

# Save voxel grid to a file
o3d.io.write_voxel_grid("saved_voxel_grid.ply", bg)
