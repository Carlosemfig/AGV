#carregar o fundo
import pickle
import MAIN as m
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
import open3d as o3d
import matplotlib.pyplot as plt
import BG_MAIN as b


# Load voxel grid from the saved file
loaded_voxel_grid = o3d.io.read_voxel_grid("bg_voxel_grid.ply")

bg=loaded_voxel_grid




"""____________________obj_________________________"""
#file_path_obj = r'C:\Users\hvendas\Desktop\agv-snr\data_dict_moving.pkl'
file_path_obj = r'data_dict_foreground.pkl'


obj=b.loadFileToArray(file_path_obj)
obj=b.remove_duplicate_points(obj)

result = b.points_outside_all_voxels(obj, bg)
pc_result=m.array_to_pc(result)
m.visualize(pc_result)
print("DONE")
print("depois",len(result))

path_file= r"subtraction_result.pcd"
m.save_pc(pc_result,path_file)
