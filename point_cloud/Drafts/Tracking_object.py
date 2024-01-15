import MAIN as m
import open3d as o3d
from sklearn.cluster import DBSCAN
import numpy as np
import time
import random
import math
import Class_object as c
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

dist=0.15
#give path, and a bg, and the size of the robot, and the space

def follow_the_path(background,path,size_x,size_y,spacing):
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    final_map=background
    tracker=c.EuclideanDistTracker3D()
    Eps=0.6
    Min_samples=10
    iteration=1

    #final_map=m.merge_arrays(global_path,background)
    for x, y, z in path:
        #criação do mapa
        cube=m.create_cubic_object((x,y),size_x,size_y,spacing)
        object=m.merge_arrays(cube,final_map)

        #adição de um obtáculo
        random_number = random.random()
        if random_number>=0.7:
            cube_2=m.create_cubic_object(m.random_coordinate_from_array(global_path),0.2,0.2,0.1)
            object=m.merge_arrays(object,cube_2)


        #detection and tracking
        result= m.subtract_array(bg,object)
        Labels, Number=m.perform_clustering(result,Eps,Min_samples)
        all, bbox= m.centroid_and_box(result,Labels,Number)
        tracker.update(bbox,dist)
        #tracker.print_stored_objects()
        print("---------------")
        print("ITERATION", iteration)
        for obj in tracker:
            #print("ID",object.get_id())
            extent=obj.get_extent()
            if extent[0]>0.3:
                obj.is_robot()
            
            obj_id = obj.get_id()
            is_robot = obj.get_robot()
            if is_robot==True:
                print(f"ID: {obj_id} is a Robot")
            else:
                print(f"ID: {obj_id} is an Obstacle")
        iteration+=1

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



