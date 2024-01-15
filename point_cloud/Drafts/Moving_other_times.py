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


def get_teta_e(trajectory,centro,matriz_de_rotação,checkpoints):
    vetor_direção=m.create_vector(trajectory[-1],trajectory[-2])
        
    #print("matriz de rotação_antes", matriz_de_rotação)
    #checkpoints= m.create_checkpoints(path,space_check)
    print(checkpoints)
    next_checkpoint=m.find_closest_checkpoint_new(centro,checkpoints)
    print("next_checkpoint",next_checkpoint)
    #tendo isto temos de avaliar a necessidade de trocar a matriz da caixa.

    matriz_de_rotação=m.change_mat_bbox(matriz_de_rotação,vetor_direção)
    #print("matriz de rotação_depois", matriz_de_rotação)
    #matriz está atualizada podemos calcular os angulos
    vetor_p= np.dot(matriz_de_rotação,np.array([1,0,0]))[:2]
    print("vetor_p",vetor_p)
    vetor_t=m.create_vector(next_checkpoint,centro)[:2]
    print("vetor_t",vetor_t)
    teta_e=math.acos((np.dot(vetor_p,vetor_t))/(np.linalg.norm(vetor_p)*np.linalg.norm(vetor_t)))
    teta_e=math.degrees(teta_e)
                    
    """teta_p=m.matrix_to_angle(matriz_de_rotação)
    teta_t=m.points_to_angle(centro,next_checkpoint)
    print("teta_t",teta_t)
    print("teta_p",teta_p)
    teta_e=teta_t-teta_p"""
    return teta_e, matriz_de_rotação
    
def next_point(teta_e,delta_t,v_max,beta,matriz_de_rotação,centro):
  pointing_direction=np.dot(matriz_de_rotação, np.array([1,0,0]))
  v=m.velocity_value(teta_e,v_max)
  next_pos=m.dist_to_pos(centro,v*delta_t,pointing_direction)
  teta_ajust=m.ajust_angle(beta,teta_e,delta_t)
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
    tracker=m.EuclideanDistTracker3D_new()
    Eps=0.6
    Min_samples=10
    space_check=10
    v_max=0.4
    delta_t=0.5
    counter=0
    position=global_path[0]
    checkpoints= m.create_checkpoints(global_path,space_check)
    #print("checkpoints",checkpoints)
    while counter<15:
      print("iteração nº:",counter)
      print("com posição",position)
      if counter==0:
        #creates a cube in the first position
        cube=m.create_cubic_object((position[0],position[1]),0.4,0.6,space)
        #detection of that cube
        object_map=m.merge_arrays(bg,cube)
        result= m.subtract_array(bg,object_map)
        Labels, Number=m.perform_clustering(result,Eps,Min_samples)
        all, bbox= m.centroid_and_box(np.array(result),Labels,Number)
        #print(bbox)
        boxes_ids_3d = tracker.update(bbox,0.4)
        for box_id_3d in boxes_ids_3d:
            #dá-nos todas as caixas que foram identificadas para cada uma das iterações
            #dá os prints para se poder ver a evolução
            cx, cy, cz, dx, dy, dz, rot_mat, obj_id = box_id_3d
            print(f"Object ID: {obj_id}, Center: ({cx}, {cy}, {cz}), Dim: ({dx}, {dy}, {dz})")
            trajectory = tracker.center_points[obj_id][3]
            print(f"  Trajectory: {trajectory}")
        center=(cx,cy,cz)
        #Find the next position
        
        next_checkpoint=m.find_closest_checkpoint_new(center,checkpoints)
        
        print("next_checkpoint", next_checkpoint)
        vetor_direção=m.create_vector(next_checkpoint,center)
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
        object=m.merge_arrays(cube,final_map)

        #Detection and tracking of the object
        result= m.subtract_array(bg,object)
        Labels, Number=m.perform_clustering(result,Eps,Min_samples)
        all, bbox= m.centroid_and_box(np.array(result),Labels,Number)
        boxes_ids_3d = tracker.update(bbox,0.4)
        for box_id_3d in boxes_ids_3d:
          #dá-nos todas as caixas que foram identificadas para cada uma das iterações
          #dá os prints para se poder ver a evolução
          cx, cy, cz, dx, dy, dz, rot_mat, obj_id = box_id_3d
          print(f"Object ID: {obj_id}, Center: ({cx}, {cy}, {cz}), Dim: ({dx}, {dy}, {dz})")
          trajectory = tracker.center_points[obj_id][3]
          print(f"  Trajectory: {trajectory}")
          #calculo de um vetor que indica a direção em que o robot está a andar.
          #if len(trajectory)>1:
            #se entrarmos neste loop significa que o robot já andou e nesse caso aplicam se as regras normais.
            #precisamos do vetor direção, centro, matriz (desatualizada), next_check.
          vetor_direção=m.create_vector(trajectory[-1],trajectory[-2])
          print("vetor direção",vetor_direção)
          centro=(cx,cy,cz)
          print("centro",centro)
          matriz_de_rotação=rot_mat
          print("matriz de rotação", matriz_de_rotação)
          teta_e, matriz_de_rotação=get_teta_e(trajectory,centro,matriz_de_rotação,checkpoints)
          print("teta_e", teta_e)
          #para calcular a próxima posição do carro:
          delta_t=0.5
          v_max=0.4
          beta=0.2
          position,ajust_angle=next_point(teta_e,delta_t,v_max,beta,matriz_de_rotação,centro)
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
