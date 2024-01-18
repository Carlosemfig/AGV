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

voxel_size = 0.3 #decreasing this value will make the subtraction process harder
treshold=0.17 #decreasing this value more will remove less of the flor.



"""_______bg______"""

file_path = r'data_dict_background.pkl'


#Load background model file to ndarray
merged_array = b.loadFileToArray(file_path)
#print(len(merged_array))

#Ndarray surface segmentation
segmentationArray = b.pointSegmentation(merged_array)

#Calculate Minimum and Maximum value points for Z,Y and Z dimensions
minAndMaxPointList = b.calculateMinAndMaxPoints(segmentationArray)

#Define the value in the Z dimension in wich all points below will be deleted
groundThresholder = minAndMaxPointList[5] + treshold

#Remove duplicate points
segmentationArrayNoDup = b.remove_duplicate_points(segmentationArray)

#Remove ground plane points
segmentationArrayNoGroundPlane = b.remove_ground_plane(segmentationArrayNoDup,groundThresholder)

##### Start Visualize point clound #####

#print(segmentationArray.size)
#print(segmentationArrayNoDup.size)

point_cloud_final=m.array_to_pc(segmentationArrayNoGroundPlane)
m.visualize(point_cloud_final)

##### End Visualize point clound #####

#Create Voxel
bg=b.voxelization(segmentationArrayNoGroundPlane,size=voxel_size)


#visualize the voxel 
o3d.visualization.draw_geometries([bg])

# Save voxel grid to a file
o3d.io.write_voxel_grid("saved_voxel_grid.ply", bg)
