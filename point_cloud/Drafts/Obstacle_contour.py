import MAIN as m
import numpy as np
import MAIN_DRIVING as d
space=0.1

global_path=m.create_straight_line((0,0),(0,10),space)


import MAIN as m
import open3d as o3d
from sklearn.cluster import DBSCAN
import numpy as np
import time
import random
import Class_object as c


#give path, and a bg, and the size of the robot, and the space

def follow_the_path(path,size_x,size_y,spacing):
    #Inicialization of the visualizer.
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()

    #Definition of the constants used:
    Eps=0.6 #Clustering
    Min_samples=10 #Clustering
    space_check=8 #Criation of the checkpoints
    v_max=0.4 #Calc of the next_pos
    delta_t=0.25 #Calc of the next_pos
    beta=1 #Calc of the next_pos
    dist=0.3
    #Inicialization of variables:
    counter=0 #Keeps track of the number of iterations performed
    position=path[0]
    ajust_angle=0
    trajectory=[]
    checkpoints= m.create_checkpoints(path,space_check)
    tracker=c.EuclideanDistTracker3D()
    flag=False
    recalc_path=False
    prev_checkpoint=None
    appear=4
    remains=20

    while counter<70:
      print("-------------")
      print("ITERATION",counter)
      if counter==appear:
         flag=True
         print("Obstacle Created")
         obstacle=m.create_cubic_object((0,3),0.2,0.2,spacing)
      elif counter==appear+remains:
         print("Obstacle Erased")
         flag=False
      cube=m.create_cubic_object((position[0],position[1]),size_x,size_y,spacing)
      map=cube

      if flag==True:
         map=m.merge_arrays(cube,obstacle)
      if ajust_angle != 0:
        #print("THE CUBE HAS ROTATED", ajust_angle)
        cube=m.rotate_cubic_object(cube,ajust_angle)
        map=cube
        if flag==True:
         map=m.merge_arrays(cube,obstacle)

      Labels, Number=m.perform_clustering(map,Eps,Min_samples)
      all, bbox= m.centroid_and_box(np.array(map),Labels,Number)
      tracker.update(bbox,dist)
      #tracker.print_stored_objects()


      for object in tracker:
        #print("ID",object.get_id())
        extent=object.get_extent()
        if extent[0]>0.3:
          object.is_robot()
          print(f"ID: {object.get_id()} is a Robot")
          trajectory=object.get_trajectory()
          next_checkpoint=m.find_closest_checkpoint_new(object.get_center(),checkpoints)
          
          
          if np.any(next_checkpoint != prev_checkpoint):
            print("New Checkpoint Reached")
            if prev_checkpoint is not None:
                 idx = np.where(np.all(path == prev_checkpoint, axis=1))[0][0]
            else:
                idx = 0

            path = path[idx:]
            prev_checkpoint = next_checkpoint
          #print ("next checkpoint",next_checkpoint)
          #print("path",path)
          if len(trajectory)<=1:
            
            #neste caso é a primeira iteração
            position,calc_angle=d.first_time(object,checkpoints,v_max,delta_t)
          
          else:
            position,calc_angle=d.other_times(object,checkpoints,delta_t,v_max,beta)
            #print("Ajustment Angle=",calc_angle)
        
        else:
           #neste caso estamos a lidar com um obstáculo
           # #é necessário verificar se está no path
           if d.is_box_in_path(object,path)==True:
            print("Box is in the Path -->Update Path")
            #necessitamos das dimensões do robot
            for obj in tracker:
                if obj.get_robot()==True:
                    robot=obj

            close_check=d.find_closest_checkpoints(object.get_center(),checkpoints)
            deviation_point=d.find_deviation_point(object,close_check,robot)
            path=d.update_path(close_check,deviation_point,path,space)
            checkpoints= m.create_checkpoints(path,space_check)
            print("Path has been Updated.")
        counter=counter+1
        ajust_angle=ajust_angle+calc_angle  
        print("Cumulative Angle=",ajust_angle)
        final=cube
        if flag==True:
            final=m.merge_arrays(cube,obstacle)
      #Visualization of this iteration
      final_array=m.merge_arrays(final, path)
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
    follow_the_path(global_path,width_cub,lenght_cub,space)

