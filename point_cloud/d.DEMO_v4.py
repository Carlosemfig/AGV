import MAIN as m
import Create_map as b

bg=b.Background
global_path=b.D1_parts
#global_path=b.Parts_T1

robot=m.create_cubic_object((27,15),0.8,0.5,0.1)
vis_robot=m.array_to_pc(robot)



#m.run_visualizer([bg,vis_robot,vis_Parts])
#m.visualize(bg)

import MAIN as m
import open3d as o3d
from sklearn.cluster import DBSCAN
import numpy as np
import time
import random
import Class_object_updates as c


   
import MAIN_DRIVING as d

def follow_the_path(bg,path,size_x,size_y,spacing):
    #Inicialization of the visualizer.
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()

    #Definition of the constants used:
    Eps=0.6 #Clustering
    Min_samples=10 #Clustering
    space_check=20 #Criation of the checkpoints
    v_max=0.5 #Calc of the next_pos
    delta_t=0.4 #Calc of the next_pos
    beta=0.9 #Calc of the next_pos
    dist=0.3
    #Inicialization of variables:
    counter=0 #Keeps track of the number of iterations performed
    position=path[0]
    ajust_angle=0
    trajectory=[]
    #checkpoints= m.create_checkpoints(path,space_check,z_offset=2)
    larger=20
    smaller=10
    checkpoints= m.create_checkpoints_with_variable_spacing(path,spacing_away=larger,spacing_near=smaller,threshold=1,z_offset=1)
    tracker=c.EuclideanDistTracker3D()
    flag=False
    prev_checkpoint=None
    appear=4
    remains=20
    path_was_recalculated=False
    obtacle_id=0
    info=(1,0)
    teta_e=0
    driving=c.Driving()
    
    mudou_o_caminho=False

    while counter<200:

      encontrou=False
      
      print("-------------")
      print("ITERATION",counter)

      print("Position",position)
      #print("Ajust angle",ajust_angle)

      #Fazer aparecer obstáculos aleatórios
      if counter==4:
        index_obstacle=1
        flag=True
        print("Obstacle Created")
        obstacle=m.create_cubic_object(path[72][:2],0.2,0.2,spacing)
        obstacle=m.delete_random_points(obstacle,0.2)
      elif counter==appear+10:
        print("Obstacle Erased")
        flag=False
      
      if counter==80:
        index_obstacle=2
        flag=True
        print("Obstacle Created")
        obstacle=m.create_cubic_object(path[210][:2],0.2,0.2,spacing)
        obstacle=m.delete_random_points(obstacle,0.2)
      elif counter==80+30:
        print("Obstacle Erased")
        flag=False
      





      #Colocar o robot na posição certa e com a rotação certa
      cube=m.create_cubic_object((position[0],position[1]),size_x,size_y,spacing)
      if ajust_angle != 0:
        #print("THE CUBE HAS ROTATED", ajust_angle)
        cube=m.rotate_cubic_object(cube,ajust_angle)
      map=cube #map é o que vai servir de apoio ao clustering
      if flag==True: #no caso de o obstáculo estar ativo
        map=m.merge_arrays(cube,obstacle)
      

      #Deteção de objectos
      
      map=m.merge_arrays(map)
      Labels, Number=m.perform_clustering(map,Eps,Min_samples)
      all, bbox= m.centroid_and_box(np.array(map),Labels,Number)
      tracker.update(bbox,dist)
      #tracker.print_stored_objects()
      is_there_an_object=False
      bbox_list=[]
      for object in tracker:
        #print("ID",object.get_id())
        extent=object.get_extent()
        
        mat=object.get_rotation_matrix()
        bbox=d.create_3d_bounding_box(object.get_center(),object.get_extent(),object.get_rotation_matrix())
        bbox_list.append(bbox)
        print("rotation matrix",mat)
        if extent[0]>0.3:
          object.is_robot()
          next_check=m.find_closest_checkpoint_new(object.get_center(),checkpoints)
          print(f"ID: {object.get_id()} is a Robot")
          # é o robot e queremos calcular a próxima posição
          trajectory=object.get_trajectory()
          if len(trajectory)<=1:
            #neste caso é a primeira iteração
            #print("ESTA É A PRIMEIRA ITERAÇÃO")
            position,calc_angle=d.first_time(object,checkpoints,v_max,delta_t,beta)
          
          else:
            #print("OUTRAS VEZES")
            position,calc_angle=d.other_times(object,teta_e,checkpoints,delta_t,v_max,beta)
        
          
        #neste caso o objecto detetado é um obstáculo
        else:
          print(f"ID: {object.get_id()} is a Obstacle")
          is_there_an_object=True
          print("index obstacle",index_obstacle)
          close_check=d.find_closest_checkpoints(object.get_center(),checkpoints,tresh=1)
          info=(object.get_id(),close_check)
          #verificar se o objecto encontrado é o mesmo que o anteriormente (base no id
          if index_obstacle==object.get_id():
            print("é o mesmo objecto")
            encontrou=True
            index_obstacle=info[0]

          if d.is_box_in_path(object,path)==True:
            print("Box is in the Path -->Update Path")
            #precisamos das dimensões do robot
            for obj in tracker:
              if obj.get_robot()==True:
                robot=obj
            close_check=d.find_closest_checkpoints(object.get_center(),checkpoints,tresh=1)
            print("CLOSE CHECK",close_check)
            info_old=(object.get_id(),close_check)
            deviation_point=d.find_deviation_point(object,close_check,robot,margin=0.5)
            path=d.update_path(close_check,deviation_point,path,spacing)
            path_was_recalculated=True
            #checkpoints= m.create_checkpoints(path,space_check,z_offset=2)
            checkpoints= m.create_checkpoints_with_variable_spacing(path,spacing_away=larger,spacing_near=smaller,threshold=1,z_offset=1)
            print("Path has been Updated.")
            mudou_o_caminho=True
          
      print("encontrou", encontrou)
      print("mudou o caminho",mudou_o_caminho)
      if encontrou==False and mudou_o_caminho==True:
        print("verificar se o caminho está livre ou não")
        print("next checkpoint",next_check)
        print(d.find_point_index(next_check,path))
        print("prev check",info_old[1][0])
        print(d.find_point_index(info_old[1][0],path))
        if d.find_point_index(next_check,path)<=d.find_point_index(info_old[1][0],path):
          print("Path should be corrected")
          path=d.reupdate_path(info_old[1],path,spacing)
          mudou_o_caminho=False
          checkpoints= m.create_checkpoints_with_variable_spacing(path,spacing_away=larger,spacing_near=smaller,threshold=1,z_offset=1)

        else:
          print("Path should not be corrected")
      counter=counter+1
      ajust_angle=ajust_angle+calc_angle  
      final=cube
      if flag==True:
        final=m.merge_arrays(cube,obstacle)
      #Visualization of this iteration
      if bbox_list:
        bbox_array=m.merge_arrays(*bbox_list)
        final=m.merge_arrays(final,bbox_array)
      final_array=m.merge_arrays(final, path,bg,checkpoints)
      point_cloud1=m.array_to_pc(final_array)
      
      visualizer.add_geometry(point_cloud1)
      visualizer.update_geometry(point_cloud1)
      visualizer.poll_events()
      visualizer.update_renderer()
      time.sleep(0.2)
      # Remove the first point cloud
      visualizer.clear_geometries()
    visualizer.run()
    visualizer.destroy_window()


width_cub=0.8
lenght_cub=0.5

if __name__ == "__main__":
    #subtract_main("bg.pcd","object_2.pcd")
    follow_the_path(bg,global_path,width_cub,lenght_cub,0.1)

