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
loaded_voxel_grid = o3d.io.read_voxel_grid("saved_voxel_grid.ply")

bg=loaded_voxel_grid


"""____________________obj_________________________"""
#file_path_obj = r'C:\Users\hvendas\Desktop\agv-snr\data_dict_moving.pkl'
file_path_obj = r'data_dict_foreground.pkl'

obj=b.loadFileToArray(file_path_obj)



obj=b.remove_duplicate_points(obj)


"""-------------data processing_________________________"""

vox_scale=b.voxel_range(bg)

arr_scale=b.array_range(obj)


resc_obj=b.rescale_array(bg,obj)





pc_obj=m.array_to_pc(resc_obj)
pc_bg=m.array_to_pc(b.voxel_centers(bg))


pc_obj.paint_uniform_color([1, 0, 0])  # Red

pc_bg.paint_uniform_color([0, 0, 1])   # Blue
# Convert voxel centers to a point cloud
#pc_voxel_extents = voxel_extents_to_pc(voxel_centers(bg), voxel_size)

# Combine the three point clouds
combined_pc = o3d.geometry.PointCloud()
combined_pc += pc_obj
combined_pc += pc_bg
#ombined_pc += pc_voxel_extents


# Visualize the overlapping
o3d.visualization.draw_geometries([combined_pc])


#visualize the voxel 
o3d.visualization.draw_geometries([bg])

#visualize the result.
print("antes",len(obj))

result = b.points_outside_all_voxels(obj, bg)
pc_result=m.array_to_pc(result)
m.visualize(pc_result)
print("DONE")
print("depois",len(result))


#path_file= r"subtraction_result.pcd"
#m.save_pc(pc_result,path_file)


