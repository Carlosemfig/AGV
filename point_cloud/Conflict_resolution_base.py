import MAIN as m
import Create_map as b
import MAIN as m
import open3d as o3d
from sklearn.cluster import DBSCAN
import numpy as np
import time
import random
import Class_object_updates as c
import MAIN_DRIVING as d

bg=b.Background
final_line=b.final_line
global_path_1=b.D1_parts
global_path_2=b.Parts_T1

def object_movement(path,object=None,position=None,ajust_angle=0,teta_e=0,Eps=0.6,Min_samples=10,trajectory=[]):
  #Colocar o robot na posição certa e com a rotação certa

  size_x=0.8
  size_y=0.5
  spacing=0.1
  
  larger=20
  smaller=10
  v_max=0.5 #Calc of the next_pos
  delta_t=0.4 #Calc of the next_pos
  beta=0.9 #Calc of the next_pos
  dist=0.3
  if position is None:
    position=path[0]
    cube=m.create_cubic_object((position[0],position[1]),size_x,size_y,spacing)
  else:
    cube=m.create_cubic_object((position[0],position[1]),size_x,size_y,spacing)

  if ajust_angle != 0:
    #print("THE CUBE HAS ROTATED", ajust_angle)
    cube=m.rotate_cubic_object(cube,ajust_angle)
  map=cube #map é o que vai servir de apoio ao clustering
  map=m.merge_arrays(map)
  Labels, Number=m.perform_clustering(map,Eps,Min_samples)
  all, bbox= m.centroid_and_box(np.array(map),Labels,Number)
  if object==None:
    object = c.Object(0,bbox[0][0] ,bbox[0][1], bbox[0][2])
  else:
    object.update_object(bbox[0][0] ,bbox[0][1], bbox[0][2])



  #print("ID",object.get_id())
  extent=object.get_extent()
  mat=object.get_rotation_matrix()
  print("rotation matrix",mat)
  checkpoints= m.create_checkpoints_with_variable_spacing(path,spacing_away=larger,spacing_near=smaller,threshold=1,z_offset=1)

  if len(trajectory)<=1:
    position,calc_angle=d.first_time(object,checkpoints,v_max,delta_t,beta)
    
  else:
      #print("OUTRAS VEZES")
    position,calc_angle=d.other_times(object,teta_e,checkpoints,delta_t,v_max,beta)
  
  ajust_angle=ajust_angle+calc_angle  
  final=cube
  #Visualization of this iteration
  final_array=m.merge_arrays(final, path,bg,checkpoints)
  point_cloud1=m.array_to_pc(final_array)
  return final_array,object,position,ajust_angle
  

def loop_visualize(visualizer,point_cloud):
  visualizer.add_geometry(point_cloud)
  visualizer.update_geometry(point_cloud)
  visualizer.poll_events()
  visualizer.update_renderer()
  time.sleep(0.2)
  # Remove the first point cloud
  visualizer.clear_geometries()
   







def follow_the_path(bg,path_1,path_2,size_x,size_y,spacing):
    #Inicialization of the visualizer.
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()

    #Definition of the constants used:
    Eps=0.6 #Clustering
    Min_samples=10 #Clustering
    #checkpoints= m.create_checkpoints(path,space_check,z_offset=2)
    larger=20
    smaller=10
    counter=0
  
    object_1=None
    position_1=None
    ajust_angle_1=0
    object_2=None
    position_2=None
    ajust_angle_2=0
    teta_e=0
    Eps=0.6
    Min_samples=10

    while counter<200:      
      point_cloud1,object_1,position_1,ajust_angle_1=object_movement(path_1,object=object_1,position=position_1,ajust_angle=ajust_angle_1,teta_e=teta_e,Eps=Eps,Min_samples=Min_samples)
      point_cloud2,object_2,position_2,ajust_angle_2=object_movement(path_2,object=object_2,position=position_2,ajust_angle=ajust_angle_2,teta_e=teta_e,Eps=Eps,Min_samples=Min_samples)
      
      #se há interseção dos raios de segurança.
      #calcular a proxima posição e o angulo de ajuste com base na situação em questão
      #são 2 robos
      #é um robot e uma caixa
      #é um robot e uma pessoa


      final_array=m.merge_arrays(point_cloud1,point_cloud2)
      point_cloud=m.array_to_pc(final_array)
      loop_visualize(visualizer,point_cloud)
      counter=counter+1
  
  

    visualizer.run()
    visualizer.destroy_window()


width_cub=0.8
lenght_cub=0.5

if __name__ == "__main__":
    #subtract_main("bg.pcd","object_2.pcd")
    follow_the_path(bg,global_path_1,global_path_2,width_cub,lenght_cub,0.1)

