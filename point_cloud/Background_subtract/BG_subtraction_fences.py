#carregar o fundo
import pickle
import MAIN as m
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
import open3d as o3d
import matplotlib.pyplot as plt
import BG_MAIN as b
import json

"""__________________ load the background structure____________"""

treshold=0.17
# Load voxel grid from the saved file
loaded_voxel_grid = o3d.io.read_voxel_grid("saved_voxel_grid.ply")

bg=loaded_voxel_grid


"""____________________obj_________________________"""
#file_path_obj = r'C:\Users\hvendas\Desktop\agv-snr\data_dict_moving.pkl'
file_path_obj = r'data_dict_foreground.pkl'

obj=b.loadFileToArray(file_path_obj)



""""________________fences________________________"""
with open('BuildingsGeojson_1.json') as json_file:
    data = json.load(json_file)

def get_2dboxes_from_json(json_name):
    with open(json_name) as json_file:
        data = json.load(json_file)

    boxes = {}
    box_id_counter = 0

    for feature in data['features']:
        coordinates = feature['geometry']['coordinates'][0]
        # Assuming the coordinates are in the format [x, y], extract x and y values
        x_values = [coord[0] for coord in coordinates]
        y_values = [coord[1] for coord in coordinates]
        
        corners= list(zip(x_values, y_values))
        corners.pop()
        boxes[box_id_counter] = corners

        # Increment box_id for the next box
        box_id_counter += 1
    return boxes
boxes=get_2dboxes_from_json('BuildingsGeojson_1.json')
print(boxes)

def transform_2d_to_3d(boxes_dict,base=-2,height=2):
    new_boxes={}
    for box_id, corners in boxes_dict.items():
        # Convert 2D corners to 3D corners

        corners_scaled = [(coord[0], coord[1]) for coord in corners]

        corners_3d = np.hstack((np.array(corners_scaled), np.ones((len(corners_scaled), 1))*base))
        corners_top = np.hstack((np.array(corners_scaled), np.ones((len(corners_scaled), 1)) * height))
        final_corners=np.vstack((corners_3d, corners_top))
        new_boxes[box_id] = final_corners
    return new_boxes

new_boxes=transform_2d_to_3d(boxes)
print("new_boxes",new_boxes)

def create_3d_bboxes(new_boxes_dict, line_color=[0.0, 1.0, 0.0]):
    geometries = []
    multiplier=1
    for box_id, corners in new_boxes_dict.items():
        # Convert 2D corners to 3D corners
        print("corners",corners)
        corners_scaled = [(coord[0] * multiplier, coord[1] * multiplier) for coord in corners]

        corners_3d = np.hstack((np.array(corners_scaled), np.ones((len(corners_scaled), 1))))
        corners_top = np.hstack((np.array(corners_scaled), np.ones((len(corners_scaled), 1))))

        # Create 3D line sets for the bounding box edges
        lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        # Set color and thickness
        line_set.colors = o3d.utility.Vector3dVector(np.tile(np.array(line_color), (len(lines), 1)))
        
        geometries.append(line_set)

    return geometries

# Assuming you have already obtained the 'boxes' dictionary
generated_geometries = create_3d_bboxes(new_boxes)

pc_obj=m.array_to_pc(obj)
pc_obj.paint_uniform_color([1, 0, 0])  # Red
combined_pc = o3d.geometry.PointCloud()
combined_pc += pc_obj

# Visualize the 3D bounding boxes
#o3d.visualization.draw_geometries(generated_geometries)
o3d.visualization.draw_geometries([combined_pc] + generated_geometries)

"""-------------data processing_________________________"""

vox_scale=b.voxel_range(bg)

arr_scale=b.array_range(obj)


resc_obj=b.rescale_array(bg,obj)




print("pc_object",resc_obj)
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



# Visualize the overlapping
#o3d.visualization.draw_geometries([combined_pc])
o3d.visualization.draw_geometries([combined_pc] + generated_geometries)


#visualize the voxel 
o3d.visualization.draw_geometries([bg])

#visualize the result.
print("antes",len(obj))

result=b.subtract_bg(obj,bg,treshold)
pc_result=m.array_to_pc(result)
m.visualize(pc_result)
print("DONE")
print("depois",len(result))


new_vis=m.array_to_pc(obj)
m.visualize(new_vis)
print("DONE")
print("depois",len(result))



new_boxes=b.get_fences('BuildingsGeojson_1.json')


result=b.final_subtraction(obj,new_boxes,bg,treshold)
pc_result=m.array_to_pc(result)
m.visualize(pc_result)
print("DONE")

path_file= r"subtraction_result.pcd"
m.save_pc(pc_result,path_file)


