import MAIN as m
import open3d as o3d
from sklearn.cluster import DBSCAN
import numpy as np
import time
import random
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
def ajust_angle(beta, teta_e, delta_t):
    """
    Gives the ajustment angle that the robot need to rotate in a time step.
    Parameters:
    beta (float): Is the constant of porporcionality for the angular velocity to be calculated.
    teta_e (float): The ajustment angle, required to achive the checkpoint in degrees.
    delta_t (float): The time corresponding to a time_step

    Returns:
    teta_ajust (float): Is the angle in degrees that the robot needs to rotate in the next iteration.
    """
    ang_vel=beta*teta_e
    teta_ajust=ang_vel*delta_t
    return(teta_ajust)

def get_teta_e(bbox,checkpoints):
    center=bbox.get_center()
    trajectory=bbox.get_trajectory()
    rot_mat=bbox.get_rotation_matrix()
    """
    Given, the checkpoints and the rot mat from the bbox, returns the teta_e.
    Parameters:
    trajectory (list): List that stores the last positions of the object.
    center(np.array): The center of the bbox (x,y,z).
    rot_mat (np.array): Rotation matrix associated with the bbox detected. 
    checkpoints (np.array): Equally spaced checkpoints of the path.

    Returns:
    teta_e (np.array): The angle (in degrees) that the robot needs to rotate to face the next checkpoint (t).
    updated_rot_mat(np.array): The updated matrix associated with the bbox detected.
    """
    #Gives the direction the car was moving, in the last iteration.
    vector_front=m.create_vector(trajectory[-1],trajectory[-2])
    
    
    #Finds the closest checkpoint to the center of the bbox detected.
    next_checkpoint=m.find_closest_checkpoint_new(center,checkpoints)
    #print("next_checkpoint",next_checkpoint)
    
    #The rot_mat of the bbox may need to be updated based on the vector_front.
    updated_rot_mat=m.change_mat_bbox(rot_mat,vector_front)
    bbox.update_rotation_matrix(updated_rot_mat)
    #The angle between the x_axis of the bbox and the x_inertial. 
    teta_p=m.matrix_to_angle(updated_rot_mat)
    #print("teta_p",teta_p)
    #The angle betwen the center_to_next_checkpoint and the x_inertial.
    teta_t=m.points_to_angle(center,next_checkpoint)
    #print("teta_t",teta_t)
    
    #The rotation that the robot needs to perform to be facing the next_checkpoint.
    teta_e=teta_t-teta_p
    return teta_e

def next_point(bbox,teta_e,delta_t,v_max,beta):
  rot_mat=bbox.get_rotation_matrix()
  center=bbox.get_center()
  """
  Given the ajustment angle and the velocity that the car moves returns the (x,y,z) of the exact position in the next time step. 
  Parameters:
  teta_e (np.array): The angle that the robot needs to rotate to face the next checkpoint (t).
  delta_t (float): The time step.
  v_max (float): The maximum value that the velocity can have.
  beta(float): Constant used to calculate the ajustment angle for each iteration.
  rot_mat(np.array): The rotation matrix associated with the bbox.
  center (np.array): The center of the detected bbox detected.


  Returns:
  next_pos(np.array): (x,y,z) position of the robot center in the next iteration.
  teta_ajust(float): The angle (in degrees) that the robot is rotating in the next iteration.
  """
  #Calculate the angles that the robot is rotating in the next iteration.
  teta_ajust=ajust_angle(beta,teta_e,delta_t)
  
  vector_p=np.dot(rot_mat, np.array([1,0,0])) [:2]
  adj= np.radians(teta_ajust)
  adjustment_matrix = np.array([[np.cos(adj), -np.sin(adj)],
                           [np.sin(adj), np.cos(adj)]])
  #Apply the value of this rotation to the x_axis of the boinding box to understand the moving direction.
  pointing_direction = np.dot(adjustment_matrix, vector_p)
  #Calculation of the velocity value based on the teta_e value.
  v=m.velocity_value(teta_e,v_max)
  print("Velocity=",v)
  
  #Calculates the next position based on the center, the velocity and the pointing direction.
  next_pos=m.dist_to_pos(center,v*delta_t,pointing_direction)
  return next_pos,teta_ajust

def first_time(bbox, checkpoints,v_max,delta_t):
  center=bbox.get_center()
  next_checkpoint=m.find_closest_checkpoint_new(center,checkpoints)
  vetor_direção=m.create_vector(next_checkpoint,center)[:2]
  dist=v_max*delta_t
  position=m.dist_to_pos(center,dist,vetor_direção)
  return(position, 0)

def other_times(bbox,checkpoints,delta_t,v_max,beta):
#Having the bbox and the trajectory is possible to calculate the next position
    teta_e=get_teta_e(bbox, checkpoints)
    #print("teta_e", teta_e)
    position,calc_angle=next_point(bbox,teta_e,delta_t,v_max,beta)
    return (position, calc_angle)

   


#give path, and a bg, and the size of the robot, and the space

def follow_the_path(bg,path,size_x,size_y,spacing):
    #Inicialization of the visualizer.
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()

    #Definition of the constants used:
    Eps=0.6 #Clustering
    Min_samples=10 #Clustering
    space_check=10 #Criation of the checkpoints
    space_check_strict=8
    v_max=0.4 #Calc of the next_pos
    delta_t=0.5 #Calc of the next_pos
    beta=1 #Calc of the next_pos
    dist=0.3
    #Inicialization of variables:
    counter=0 #Keeps track of the number of iterations performed
    position=path[0]
    ajust_angle=0
    trajectory=[]
    checkpoints= m.create_checkpoints_with_variable_spacing(path,spacing_away=space_check,spacing_near=space_check_strict)
    tracker=c.EuclideanDistTracker3D()
    

    while counter<50:
      cube=m.create_cubic_object((position[0],position[1]),size_x,size_y,spacing)
      if ajust_angle != 0:
        #print("THE CUBE HAS ROTATED", ajust_angle)
        cube=m.rotate_cubic_object(cube,ajust_angle)
      
      Labels, Number=m.perform_clustering(cube,Eps,Min_samples)
      all, bbox= m.centroid_and_box(np.array(cube),Labels,Number)
      tracker.update(bbox,dist)
      #tracker.print_stored_objects()
      for object in tracker:
        #print("ID",object.get_id())
        extent=object.get_extent()
        if extent[0]>0.1:
          # é o robot e queremos calcular a próxima posição
          trajectory=object.get_trajectory()
          if len(trajectory)<=1:
            #neste caso é a primeira iteração
            position,calc_angle=first_time(object,checkpoints,v_max,delta_t)
          
          else:
            print("---------------------")
            print("ITERATION",counter)
            position,calc_angle=other_times(object,checkpoints,delta_t,v_max,beta)
            print("Ajustment Angle=",calc_angle)
        
        counter=counter+1
        ajust_angle=ajust_angle+calc_angle
        print("Cumulative Angle=",ajust_angle)  
      
      #Visualization of this iteration
      final_array=m.merge_arrays(cube, path,checkpoints)
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

