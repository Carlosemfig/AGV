import MAIN as m
import open3d as o3d
from sklearn.cluster import DBSCAN
import numpy as np
import time
import random
import math

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
#line1_start = (1.0, 0.0)
line1_start = (1.0, 5.0)
line1_end = (1.0, 6.0)
line2_start = (1.0, 6.0)
line2_end = (4.0, 6.0)
line3_start = (4.0, 6.0)
line3_end = (4.0, 12.0)
path1 =m.create_straight_line(line1_start, line1_end, space)
path2=m.create_straight_line(line2_start,line2_end,space)
path3=m.create_straight_line(line3_start,line3_end,space)
global_path= m.merge_arrays(path1,path2)
global_path=m.merge_arrays(global_path,path3)

final_map=m.merge_arrays(global_path,bg)

iterations=[0,1,2,3,4,5]




#give path, and a bg, and the size of the robot, and the space

def follow_the_path(bg,path,size_x,size_y,spacing):
    #Inicialization of the visualizer.
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()

    #Definition of the constants used:
    Eps=0.6 #Clustering
    Min_samples=10 #Clustering
    space_check=8 #Criation of the checkpoints
    v_max=0.4 #Calc of the next_pos
    delta_t=0.5 #Calc of the next_pos
    beta=1 #Calc of the next_pos

    #Inicialization of variables:
    counter=0 #Keeps track of the number of iterations performed
    position=path[0]
    ajust_angle=0
    trajectory=[]
    checkpoints= m.create_checkpoints(path,space_check)

    while counter<50:
      print("ITERAÇÃO nº",counter)
      #This if is used for the start of the robot
      if counter==0:
        # 1- Initializes the cube in the first position of the path
        cube=m.create_cubic_object((position[0],position[1]),size_x,size_y,spacing)
        
        #Detection of that cube in the map, returns a bbox (dict)used from now on
        Labels, Number=m.perform_clustering(cube,Eps,Min_samples)
        all, bbox= m.centroid_and_box(np.array(cube),Labels,Number)
        id_key = next(iter(bbox))
        center=bbox[0][0]
        extent=bbox[0][1]
        #print(f"Object ID: {id_key}, Center: ({center}), Dim: ({extent})")
        rot_mat=bbox[0][2]
        trajectory.append(center)
        #print(f"  Trajectory: {trajectory}")

        #Chosing the next iteration
        #For the start it will move forward to the next checkpoint at v max. That will be the front.
        next_checkpoint=m.find_closest_checkpoint_new(center,checkpoints)
        vetor_direção=m.create_vector(next_checkpoint,center)[:2]
        dist=v_max*delta_t
        position=m.dist_to_pos(center,dist,vetor_direção)

        #Update of the variables inicialized
        counter=counter+1

        #Visualization of this iteration
        final_array=m.merge_arrays(cube, path)
        point_cloud1=m.array_to_pc(final_array)
        visualizer.add_geometry(point_cloud1)
        visualizer.update_geometry(point_cloud1)
        visualizer.poll_events()
        visualizer.update_renderer()
        time.sleep(0.5)
        # Remove the first point cloud
        visualizer.clear_geometries()


      else:
        #From the second iteration forward it is possible to ajust the rot_mat.
        #And therefore it is possible to calculate the next pos, based on the teta_e.
  
        #1 - Place the cube in the map, rotate acoardingly with the rotation angle.
        cube=m.create_cubic_object((position[0],position[1]),size_x,size_y,spacing)
        if ajust_angle != 0:
          print("THE CUBE HAS ROTATED", ajust_angle)
          cube=m.rotate_cubic_object(cube,ajust_angle)
    
        #Detection of the object using the bbox, and trajectory.
        Labels, Number=m.perform_clustering(cube,Eps,Min_samples)
        all, bbox= m.centroid_and_box(np.array(cube),Labels,Number)
        print("TESTE Nº1", bbox)
        first_key = next(iter(bbox))
        center=bbox[0][0]
        extent=bbox[0][1]
        #print(f"Object ID: {first_key}, Center: ({center}), Dim: ({extent})")
        rot_mat=bbox[0][2]
        #print("Rotation matrix", rot_mat)
        trajectory.append(center)
        #print(f"  Trajectory: {trajectory}")

        #Having the bbox and the trajectory is possible to calculate the next position
        teta_e, rot_mat=m.get_teta_e(trajectory,center,rot_mat,checkpoints)
        print("teta_e", teta_e)
        position,calc_angle=m.next_point(teta_e,delta_t,v_max,beta,rot_mat,center)
        
        #Update the variables (position was updated above)
        ajust_angle=ajust_angle+calc_angle
        counter=counter+1
        print("This iteration ajustment angle",calc_angle)
        print("Next Position", position)


        #Visualization of this iteration
        final_array=m.merge_arrays(cube,path,np.array(trajectory))
        point_cloud1=m.array_to_pc(final_array)
        visualizer.add_geometry(point_cloud1)
        visualizer.update_geometry(point_cloud1)
        visualizer.poll_events()
        visualizer.update_renderer()
        time.sleep(0.5)
        # Remove the first point cloud
        visualizer.clear_geometries()
    visualizer.run()
    visualizer.destroy_window()


width_cub=0.4
lenght_cub=0.6

if __name__ == "__main__":
    #subtract_main("bg.pcd","object_2.pcd")
    follow_the_path(bg,global_path,width_cub,lenght_cub,space)

