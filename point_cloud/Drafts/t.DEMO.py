import MAIN as m
import Create_map as b

bg=b.Background
global_path=b.D1_parts
#global_path=b.Parts_T1

robot=m.create_cubic_object((27,15),0.8,0.5,0.1)
vis_robot=m.array_to_pc(robot)

obj1=m.create_cubic_object((2,1),0.3,0.3,0.1)
obj2=m.create_cubic_object((1,0.5),0.3,0.3,0.1)
obj3=m.create_cubic_object((3,0.5),0.3,0.3,0.1)
obj4=m.create_cubic_object((1,1.5),0.3,0.3,0.1)
obj5=m.create_cubic_object((3,1.5),0.3,0.3,0.1)
Parts=m.merge_arrays(obj1,obj2,obj3,obj4,obj5)
vis_Parts=m.array_to_pc(Parts)

#m.run_visualizer([bg,vis_robot,vis_Parts])
#m.visualize(bg)

import MAIN as m
import open3d as o3d
from sklearn.cluster import DBSCAN
import numpy as np
import time
import random
import Class_object as c




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
    #print("vetor frente",vector_front)
    #Finds the closest checkpoint to the center of the bbox detected.
    next_checkpoint=m.find_closest_checkpoint_new(center,checkpoints)
    print("next checkpoint",next_checkpoint)
    #print("next_checkpoint",next_checkpoint)
    
    #The rot_mat of the bbox may need to be updated based on the vector_front.
    updated_rot_mat=m.change_mat_bbox(rot_mat,vector_front)
    bbox.update_rotation_matrix(updated_rot_mat)
    #The angle between the x_axis of the bbox and the x_inertial. 
    teta_p=m.matrix_to_angle(updated_rot_mat)
    teta_p=m.normalize_angle(teta_p)
    #print("teta_p",teta_p)
    #The angle betwen the center_to_next_checkpoint and the x_inertial.
    teta_t=m.points_to_angle(center,next_checkpoint)
    teta_t=m.normalize_angle(teta_t)
    #print("teta_t",teta_t)
    
    #The rotation that the robot needs to perform to be facing the next_checkpoint.
    teta_e=teta_t-teta_p
    #print("teta_e",teta_e)
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
  #print("vector p", vector_p)
  adj= np.radians(teta_ajust)
  adjustment_matrix = np.array([[np.cos(adj), -np.sin(adj)],
                           [np.sin(adj), np.cos(adj)]])
  #Apply the value of this rotation to the x_axis of the boinding box to understand the moving direction.
  pointing_direction = np.dot(adjustment_matrix, vector_p)
  #print("Point direction",pointing_direction)
  #Calculation of the velocity value based on the teta_e value.
  v=m.velocity_value(teta_e,v_max)
  #Calculates the next position based on the center, the velocity and the pointing direction.
  next_pos=m.dist_to_pos(center,v*delta_t,pointing_direction)
  return next_pos,teta_ajust


def first_time_non_col(bbox, checkpoints,v_max,delta_t,beta):
  center=bbox.get_center()
  rot_mat=bbox.get_rotation_matrix()
  next_checkpoint=m.find_closest_checkpoint_new(center,checkpoints)
  vector_front=m.create_vector(next_checkpoint,center)[:2]
  # Add a zero at the end
  vector_front = np.append(vector_front, 0) 
  #print("VETOR_FRENTE",vector_front)

  #The rot_mat of the bbox may need to be updated based on the vector_front.
  updated_rot_mat=m.change_mat_bbox(rot_mat,vector_front)
  bbox.update_rotation_matrix(updated_rot_mat)
  #The angle between the x_axis of the bbox and the x_inertial. 
  teta_p=m.matrix_to_angle(updated_rot_mat)
  teta_p=m.normalize_angle(teta_p)
  #print("teta_p",teta_p)
  #The angle betwen the center_to_next_checkpoint and the x_inertial.
  teta_t=m.points_to_angle(center,next_checkpoint)
  teta_t=m.normalize_angle(teta_t)
  #print("teta_t",teta_t)
  
  #The rotation that the robot needs to perform to be facing the next_checkpoint.
  teta_e=teta_t-teta_p
  #print("TETA_E",teta_e)
  
  position,calc_angle=next_point(bbox,teta_e,delta_t,v_max,beta)
  return (position, calc_angle)


def other_times(bbox,checkpoints,delta_t,v_max,beta):
#Having the bbox and the trajectory is possible to calculate the next position
    teta_e=get_teta_e(bbox, checkpoints)
    #print("teta_e", teta_e)
    position,calc_angle=next_point(bbox,teta_e,delta_t,v_max,beta)
    return (position, calc_angle)

   

def create_checkpoints(path, min_spacing, max_density_spacing=5, z_offset=0.0):
    """
    Creates checkpoints along the given path with variable density.

    Parameters:
    path (np array): The path represented as a numpy array.
    min_spacing (float): The minimum spacing between checkpoints.
    max_density_spacing (float): The maximum spacing allowed in dense areas.
    z_offset (float, optional): The offset in the z-direction. Defaults to 0.0.

    Returns:
    checkpoints (np array): Numpy array representing checkpoints along the path with variable density.
    """
    # Ensure that the path is a numpy array
    path = np.array(path)
    
    checkpoints = []
    
    # Calculate distances from each point to the corners (you need to replace the corner coordinates)
    corner_distances = np.min(np.abs(path[:, :2] - np.array([[0, 0], [0, 1], [1, 0], [1, 1]])), axis=1)

    # Normalize distances to the range [0, 1]
    normalized_distances = (corner_distances - np.min(corner_distances)) / (np.max(corner_distances) - np.min(corner_distances))

    # Calculate variable spacing based on normalized distances
    spacing = min_spacing + normalized_distances * (max_density_spacing - min_spacing)

    # Iterate over the path with the specified variable spacing
    for i in range(0, len(path)):
        checkpoint = path[i].copy()  # Create a copy to avoid modifying the original path
        checkpoint[2] += z_offset    # Apply the z-offset
        checkpoints.append(checkpoint)
    
    return np.array(checkpoints)
   
import MAIN_DRIVING as d
#give path, and a bg, and the size of the robot, and the space

def find_point_index(target_point, point_array):
    """
    Find the index of the nearest point in the array to the target point.

    Parameters:
    target_point (tuple or list): The target point (x, y).
    point_array (np array): Numpy array containing points (each row is a point, columns are x and y).

    Returns:
    nearest_index (int): Index of the nearest point in the array.
    """
    target_point = np.array(target_point)
    point_array = np.array(point_array)

    # Calculate the Euclidean distance between the target point and all points in the array
    distances = np.linalg.norm(point_array - target_point, axis=1)

    # Find the index of the point with the minimum distance
    nearest_index = np.argmin(distances)

    return nearest_index

def follow_the_path(bg,path,size_x,size_y,spacing):
    #Inicialization of the visualizer.
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()

    #Definition of the constants used:
    Eps=0.6 #Clustering
    Min_samples=10 #Clustering
    space_check=20 #Criation of the checkpoints
    v_max=0.4 #Calc of the next_pos
    delta_t=0.4 #Calc of the next_pos
    beta=1 #Calc of the next_pos
    dist=0.3
    #Inicialization of variables:
    counter=0 #Keeps track of the number of iterations performed
    position=path[0]
    ajust_angle=0
    trajectory=[]
    #checkpoints= m.create_checkpoints(path,space_check,z_offset=2)
    checkpoints= m.create_checkpoints_with_variable_spacing(path,spacing_away=30,spacing_near=8,threshold=1,z_offset=1)
    tracker=c.EuclideanDistTracker3D()
    flag=False
    prev_checkpoint=None
    appear=4
    remains=20
    path_was_recalculated=False
    
    info=(1,0)
    index_obstacle=1
    mudou_o_caminho=False

    while counter<200:

      encontrou=False
      
      print("-------------")
      print("ITERATION",counter)

      print("Position",position)
      #print("Ajust angle",ajust_angle)

      #Fazer aparecer obstáculos aleatórios
      if counter==appear:
        flag=True
        print("Obstacle Created")
        obstacle=m.create_cubic_object(path[70][:2],0.2,0.2,spacing)
      elif counter==appear+remains:
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
      Labels, Number=m.perform_clustering(map,Eps,Min_samples)
      all, bbox= m.centroid_and_box(np.array(map),Labels,Number)
      tracker.update(bbox,dist)
      tracker.print_stored_objects()
      is_there_an_object=False
      for object in tracker:
        #print("ID",object.get_id())
        extent=object.get_extent()
        if extent[0]>0.3:
          object.is_robot()
          next_check=m.find_closest_checkpoint_new(object.get_center(),checkpoints)
          print(f"ID: {object.get_id()} is a Robot")
          # é o robot e queremos calcular a próxima posição
          trajectory=object.get_trajectory()
          if len(trajectory)<=1:
            #neste caso é a primeira iteração
            #print("ESTA É A PRIMEIRA ITERAÇÃO")
            position,calc_angle=first_time_non_col(object,checkpoints,v_max,delta_t,beta)
          
          else:
            #print("OUTRAS VEZES")
            position,calc_angle=other_times(object,checkpoints,delta_t,v_max,beta)
        
          
        #neste caso o objecto detetado é um obstáculo
        else:
          print(f"ID: {object.get_id()} is a Obstacle")
          is_there_an_object=True
          #verificar se o objecto encontrado é o mesmo que o anteriormente (base no id)

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
            close_check=d.find_closest_checkpoints(object.get_center(),checkpoints)
            info=(object.get_id(),close_check)
            deviation_point=d.find_deviation_point(object,close_check,robot,margin=0.5)
            path=d.update_path(close_check,deviation_point,path,spacing)
            path_was_recalculated=True
            #checkpoints= m.create_checkpoints(path,space_check,z_offset=2)
            checkpoints= m.create_checkpoints_with_variable_spacing(path,spacing_away=30,spacing_near=8,threshold=1,z_offset=1)
            print("Path has been Updated.")
            mudou_o_caminho=True
          
      print("encontrou", encontrou)
      print("mudou o caminho",mudou_o_caminho)
      if encontrou==False and mudou_o_caminho==True:
        print("verificar se o caminho está livre ou não")
        print("next checkpoint",next_check)
        print(find_point_index(next_check,path))
        print("prev check",info[1][0])
        print(find_point_index(info[1][0],path))
        if find_point_index(next_check,path)<=find_point_index(info[1][0],path):
          print("Path should be corrected")
          path=reupdate_path(info[1],path,spacing)
          mudou_o_caminho=False
          checkpoints= m.create_checkpoints_with_variable_spacing(path,spacing_away=30,spacing_near=8,threshold=1,z_offset=1)

        else:
          print("Path should not be corrected")
      counter=counter+1
      ajust_angle=ajust_angle+calc_angle  
      final=cube
      if flag==True:
        final=m.merge_arrays(cube,obstacle)
      #Visualization of this iteration
      final_array=m.merge_arrays(final, path,bg,Parts,checkpoints)
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

def reupdate_path(closest_check,original_path,spacing):
  prev_index = find_point_index(closest_check[0],original_path)
  next_index= find_point_index(closest_check[1],original_path)
  previous_path = (original_path[:prev_index])
  next_path=(original_path[next_index + 1:])
  line=m.create_straight_line(closest_check[0][:2], closest_check[1][:2], spacing)
  new_path=m.merge_arrays(previous_path,line,next_path)
  return(new_path)

width_cub=0.8
lenght_cub=0.5

if __name__ == "__main__":
    #subtract_main("bg.pcd","object_2.pcd")
    follow_the_path(bg,global_path,width_cub,lenght_cub,0.1)

