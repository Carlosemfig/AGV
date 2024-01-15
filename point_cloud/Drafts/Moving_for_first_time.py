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


def dist_to_pos(current_position, distance, pointing_direction):
    """
    Given the distance of the next position, it returns the coordinates of the next position.

    Parameters:
    current_position (np.array): Coordinates (x, y, z) of the current position where the robot is placed.
    distance (float): Distance that the next position is from the current_position.
    pointing_direction (np.array): Normalized direction vector in which the robot is pointing.

    Returns:
    next_position (np.array): Coordinates (x, y, z) of the next position where to place the robot.
    """
    # Calculate the next position based on the pointing direction and distance
    next_position = current_position + distance * pointing_direction
    
    
    return next_position
   

def velocity_value(teta_e,max_value):
    """
    Gives a value of the velocity of the robot based on the ajustment angle.
    Parameters:
    teta_e (float): The ajustment angle value in degrees.
    max_value (float): The maximum velocity, used when the robot is moving forward.

    Returns:
    velocity (float): The value of velocity used in that timestep, to calculate the distance to move. 
    """
    teta_e=math.radians(teta_e)
    velocity=max_value*math.cos(teta_e)
    return velocity

Eps=0.6
Min_samples=10
spacing_between_checkpoints=10

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
  next_position=dist_to_pos(center,dist,vetor_direção)
  print("next_position", next_position)
  return next_position





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
checkpoints= m.create_checkpoints(global_path,spacing_between_checkpoints)
next_checkpoint=m.find_closest_checkpoint(center,checkpoints)
print("next_checkpoint", next_checkpoint)

#4 - Calcular o vetor direção (normalizado) na qual o robot vai andar
vetor_direção=m.create_vector(next_checkpoint,center)

#5 - Atribuir uma velocidade inicial ao robot, para a primeira iteração.
v_max=0.4

#6 - Calcular a próxima posição do robot dando a pos_atual, a distancia e o vetor_direção
delta_t=0.5
dist=v_max*delta_t
next_position=dist_to_pos(center,dist,vetor_direção)
print("next_position", next_position)




