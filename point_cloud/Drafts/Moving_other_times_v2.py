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
line3_end = (4.0, 0.0)
path1 =m.create_straight_line(line1_start, line1_end, space)
path2=m.create_straight_line(line2_start,line2_end,space)
path3=m.create_straight_line(line3_start,line3_end,space)
global_path= m.merge_arrays(path1,path2)
global_path=m.merge_arrays(global_path,path3)

final_map=m.merge_arrays(global_path,bg)

iterations=[0,1,2,3,4,5]


def get_teta_e(trajectory,centro,matriz_de_rotação,checkpoints):
    vetor_direção=m.create_vector(trajectory[-1],trajectory[-2])
        
    #print("matriz de rotação_antes", matriz_de_rotação)
    #checkpoints= m.create_checkpoints(path,space_check)
    #print(checkpoints)
    next_checkpoint=m.find_closest_checkpoint_new(centro,checkpoints)
    print("next_checkpoint",next_checkpoint)
    #tendo isto temos de avaliar a necessidade de trocar a matriz da caixa.

    matriz_de_rotação=m.change_mat_bbox(matriz_de_rotação,vetor_direção)
    #print("matriz de rotação_depois", matriz_de_rotação)
    #matriz está atualizada podemos calcular os angulos
    """vetor_p= np.dot(matriz_de_rotação,np.array([1,0,0]))[:2]
    print("vetor_p",vetor_p)
    vetor_t=m.create_vector(next_checkpoint,centro)[:2]
    print("vetor_t",vetor_t)
    teta_e=math.acos((np.dot(vetor_t,vetor_p))/(np.linalg.norm(vetor_p)*np.linalg.norm(vetor_t)))
    teta_e=math.degrees(teta_e)"""
                    
    teta_p=m.matrix_to_angle(matriz_de_rotação)
    teta_t=m.points_to_angle(centro,next_checkpoint)
    print("teta_t",teta_t)
    print("teta_p",teta_p)
    teta_e=teta_t-teta_p
    return teta_e, matriz_de_rotação
    
def next_point(teta_e,delta_t,v_max,beta,matriz_de_rotação,centro):
  
  teta_ajust=m.ajust_angle(beta,teta_e,delta_t)
  vector_p=np.dot(matriz_de_rotação, np.array([1,0,0])) [:2]
  adj= np.radians(teta_ajust)
  rotation_matrix = np.array([[np.cos(adj), -np.sin(adj)],
                           [np.sin(adj), np.cos(adj)]])
  # Apply the rotation to vector p
  pointing_direction = np.dot(rotation_matrix, vector_p)
  print("this is the pointing vector",pointing_direction)
  v=m.velocity_value(teta_e,v_max)
  print("velocity value",v)
  next_pos=m.dist_to_pos(centro,v*delta_t,pointing_direction)
  return next_pos,teta_ajust


def mov_first_time(global_path,bg,Eps,Min_samples,space_check):
    #1 - criar o cubo e colocá-lo no mapa
  first_pos=global_path[0]
  print("posição inicial do cubo",(first_pos[0],first_pos[1]))
  cube=m.create_cubic_object((first_pos[0],first_pos[1]),0.4,0.6,space)
  object_map=m.merge_arrays(bg,cube)
    #este é o mapa com o objecto colocado

  #2 - identificar o cubo no mapa
  result= m.subtract_array(bg,object_map)
  Labels, Number=m.perform_clustering(result,Eps,Min_samples)
  all, bbox= m.centroid_and_box(np.array(result),Labels,Number)

  #3 - Para passar à próxima fase precisamos do centro e do próximo checkpoint.
  center=bbox[0][0]
  print("center", center)
  checkpoints= m.create_checkpoints(global_path,space_check)
  next_checkpoint=m.find_closest_checkpoint(center,checkpoints)
  print("next_checkpoint", next_checkpoint)

  #4 - Calcular o vetor direção (normalizado) na qual o robot vai andar
  vetor_direção=m.create_vector(next_checkpoint,center)

  #5 - Atribuir uma velocidade inicial ao robot, para a primeira iteração.
  v_max=0.4

  #6 - Calcular a próxima posição do robot dando a pos_atual, a distancia e o vetor_direção
  delta_t=0.5
  dist=v_max*delta_t
  next_position=m.dist_to_pos(center,dist,vetor_direção)
  print("next_position", next_position)
  return next_position




#give path, and a bg, and the size of the robot, and the space

def follow_the_path(background,path,size_x,size_y,spacing):
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    final_map=background
    #tracker=m.EuclideanDistTracker3D_new()
    Eps=0.6
    Min_samples=10
    space_check=5
    v_max=0.4
    delta_t=0.5
    counter=0
    position=global_path[0]
    ajust_angle=0
    trajectory=[]
    checkpoints= m.create_checkpoints(global_path,space_check)
    #print("checkpoints",checkpoints)
    while counter<50:
      print("iteração nº:",counter)
      #print("com posição",position)
      if counter==0:
        #creates a cube in the first position
        cube=m.create_cubic_object((position[0],position[1]),0.4,0.6,space)
        #detection of that cube
        object_map=m.merge_arrays(bg,cube)
        result= m.subtract_array(bg,object_map)
        Labels, Number=m.perform_clustering(result,Eps,Min_samples)
        all, bbox= m.centroid_and_box(np.array(result),Labels,Number)
        first_key = next(iter(bbox))
        center=bbox[0][0]
        extent=bbox[0][1]
        #print(f"Object ID: {first_key}, Center: ({center}), Dim: ({extent})")
        rot_mat=bbox[0][2]
        trajectory.append(center)
        #print(f"  Trajectory: {trajectory}")
        #Find the next position
        
        next_checkpoint=m.find_closest_checkpoint_new(center,checkpoints)
        
        #print("next_checkpoint", next_checkpoint)
        vetor_direção=m.create_vector(next_checkpoint,center)[:2]
        dist=v_max*delta_t
        position=m.dist_to_pos(center,dist,vetor_direção)
        counter=counter+1

        point_cloud1=m.array_to_pc(object_map)
        visualizer.add_geometry(point_cloud1)
        visualizer.update_geometry(point_cloud1)
        visualizer.poll_events()
        visualizer.update_renderer()
        time.sleep(0.5)
        # Remove the first point cloud
        visualizer.clear_geometries()


      else:
        #Place the cube in the map
        cube=m.create_cubic_object((position[0],position[1]),0.4,0.6,space)
        if ajust_angle != 0:
          print("THE CUBE HAS ROTATED", ajust_angle)
          cube=m.rotate_cubic_object(cube,ajust_angle)
        
        #merge the resulting cube either is the rotated or not.
        object=m.merge_arrays(cube,final_map)

        #Detection and tracking of the object
        #result= m.subtract_array(bg,object)
        Labels, Number=m.perform_clustering(cube,Eps,Min_samples)
        all, bbox= m.centroid_and_box(np.array(cube),Labels,Number)
        #print("TESTE Nº1", bbox)
        first_key = next(iter(bbox))
        center=bbox[0][0]
        extent=bbox[0][1]
        #print(f"Object ID: {first_key}, Center: ({center}), Dim: ({extent})")
        rot_mat=bbox[0][2]
        #print("Rotation matrix", rot_mat)
        trajectory.append(center)
        #print(f"  Trajectory: {trajectory}")
        vetor_direção=m.create_vector(trajectory[-1],trajectory[-2])
        #print("vetor direção",vetor_direção)
        teta_e, rot_mat=get_teta_e(trajectory,center,rot_mat,checkpoints)
        print("teta_e", teta_e)
        #para calcular a próxima posição do carro:
        delta_t=0.5
        v_max=0.4
        beta=0.4
        position,calc_angle=next_point(teta_e,delta_t,v_max,beta,rot_mat,center)
        ajust_angle=ajust_angle+calc_angle
        print("This iteration ajustment angle",calc_angle)
        #print("Next ajust an", ajust_angle)
        print("Next Position", position)
        counter=counter+1
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


width_cub=0.4
lenght_cub=0.6

if __name__ == "__main__":
    #subtract_main("bg.pcd","object_2.pcd")
    follow_the_path(bg,global_path,width_cub,lenght_cub,space)

