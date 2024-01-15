import MAIN as m
import open3d as o3d
from sklearn.cluster import DBSCAN
import numpy as np
import time

# Dar load do backgorund
bg=m.load_pc("bg.pcd")
bg=m.pc_to_array(bg)


# Example usage of the function
center_cub=(1,1)
width_cub = 0.6  # Width range
length_cub = 0.6  # Length range
height_cub = 0.6  # Height of the cubic object
space=0.1

# Define the starting and ending coordinates of the three straight lines
line1_start = (1.0, 0.0)
line1_end = (1.0, 6.0)
line2_start = (1.0, 6.0)
line2_end = (4.0, 6.0)
line3_start = (4.0, 6.0)
line3_end = (4.0, 0.0)
path1 =m.create_straight_line(line1_start, line1_end, space)
path2=m.create_straight_line(line2_start,line2_end,space)
path3=m.create_straight_line(line3_start,line3_end,space)
global_path= m.merge_arrays(path1,path2)
global_path=m.merge_arrays(global_path,path3)

final_map=m.merge_arrays(global_path,bg)

iterations=[0,1,2,3,4,5]


#give path, and a bg, and the size of the robot, and the space

def follow_the_path(background,path,size_x,size_y,spacing):
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    final_map=background
    tracker=m.EuclideanDistTracker3D_new_new()
    Eps=0.6
    Min_samples=10

    #final_map=m.merge_arrays(global_path,background)
    for x, y, z in path:
        #criação do mapa
        cube=m.create_cubic_object((x,y),0.4,0.6,spacing)
        object=m.merge_arrays(cube,final_map)
        
        #detection and tracking
        result= m.subtract_array(bg,object)
        Labels, Number=m.perform_clustering(result,Eps,Min_samples)
        all, bbox= m.centroid_and_box(np.array(result),Labels,Number)
        boxes_ids_3d = tracker.update(bbox,0.2)
        for box_id_3d in boxes_ids_3d:
            cx, cy, cz, dx, dy, dz, obj_id = box_id_3d
            print(f"Object ID: {obj_id}, Center: ({cx}, {cy}, {cz}), Dim: ({dx}, {dy}, {dz})")
            trajectory = tracker.center_points[obj_id][2]
            print(f"  Trajectory: {trajectory}")

            #calculo de um vetor que indica a direção em que o robot está a andar.
            if len(trajectory)>1:
                current_p=trajectory[-1]
                previous_p=trajectory[-2]
                vector=m.create_vector(current_p,previous_p)
                print(vector)
            
        #visualização
        point_cloud1=m.array_to_pc(object)
        visualizer.add_geometry(point_cloud1)
        visualizer.update_geometry(point_cloud1)
        visualizer.poll_events()
        visualizer.update_renderer()
        time.sleep(0.5)
        # Remove the first point cloud
        visualizer.clear_geometries()
    visualizer.run()
    visualizer.destroy_window()

width_cub=0.6
lenght_cub=0.4

if __name__ == "__main__":
    #subtract_main("bg.pcd","object_2.pcd")
    follow_the_path(bg,global_path,width_cub,lenght_cub,space)



