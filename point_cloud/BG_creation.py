#carregar o fundo
import pickle
import MAIN as m
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
import open3d as o3d
import matplotlib.pyplot as plt

voxel_size = 0.3


"""_______bg______"""
#carregamos os pontos
file_path = r'C:\Users\hvendas\Desktop\agv-snr\data_dict_moving_2.pkl'
with open(file_path, 'rb') as file:
    loaded_data_dict = pickle.load(file)

#passamos para array  
num_arrays_to_merge =99
i=0
merged_array = []

for key, value in loaded_data_dict.items():
    if i < num_arrays_to_merge:
        merged_array.append(value)
        #print("merged array",merged_array)
        i=i+1
merged_array = np.concatenate(merged_array, axis=0)

def voxelization(array,size=0.01):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(array)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=size)
    return (voxel_grid)

bg=voxelization(merged_array,size=voxel_size)

import open3d as o3d

# Save voxel grid to a file
o3d.io.write_voxel_grid("saved_voxel_grid.ply", bg)