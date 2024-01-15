import numpy as np
import math
import MAIN as m
import MAIN_DRIVING as d


class Object:
    """
    Represents an object in 3D space with an ID, center, extent, rotation matrix, and trajectory.

    Attributes:
    - id (int): The unique identifier for the object.
    - center (list or tuple): The center coordinates of the object.
    - extent (list or tuple): The extent or size of the object.
    - rotation_matrix (list or np.array): The rotation matrix representing the orientation of the object.
    - trajectory (list of lists): The trajectory of the object, represented as a list of coordinates.

    Methods:
    - __init__: Initialize a new object with the given ID, center, extent, and rotation matrix.
    - update_object: Update the characteristics of the object. The ID remains unchanged.
    - update_rotation_matrix: Update only the rotation matrix of the object.
    - update_trajectory: Update only the trajectory of the object.
    - get_id: Get the unique identifier of the object.
    - get_rotation_matrix: Get the rotation matrix of the object.
    - get_center: Get the center coordinates of the object.
    - get_trajectory: Get the trajectory of the object.
    - get_extent: Get the extent or size of the object.
    - __str__: Return a string representation of the object.
    """
    def __init__(self, obj_id, center, extent, rotation_matrix):
        """
        Initialize a new object with the given ID, center, extent, and rotation matrix.

        Parameters:
        - obj_id (int): The unique identifier for the object.
        - center (list or tuple): The center coordinates of the object.
        - extent (list or tuple): The extent or size of the object.
        - rotation_matrix (list or np.array): The rotation matrix representing the orientation of the object.

        The object is also initialized with an empty trajectory containing the initial center coordinates.
        """
        self.id = obj_id
        self.robot=False
        self.center = center
        self.extent = extent
        self.rotation_matrix = np.array(rotation_matrix)
        self.trajectory = [[center[0], center[1], center[2]]]
        self.vector_front=None
        self.velocity=None

    def is_robot(self):
        self.robot=True
    


    def update_vector_front(self,vector_f):
        self.vector_front=vector_f
    
    def get_vector_front(self):
        return self.vector_front

    def update_velocity(self,vel):
        self.velocity=vel

    def get_velocity(self):
        return self.velocity

    def update_object(self, center, extent, rotation_matrix):
        """
        Update the characteristics of the object. The ID remains unchanged.

        Parameters:
        - center (list or tuple): The new center coordinates of the object.
        - extent (list or tuple): The new extent or size of the object.
        - rotation_matrix (list or np.array): The new rotation matrix representing the orientation of the object.

        The trajectory is updated with the new center coordinates.
        """
        self.center = center
        self.extent = extent
        self.rotation_matrix = np.array(rotation_matrix)
        self.trajectory.append([center[0], center[1], center[2]])

    def update_rotation_matrix(self, new_rotation_matrix):
        """
        Update only the rotation matrix of the object.

        Parameters:
        - new_rotation_matrix (list or np.array): The new rotation matrix representing the orientation of the object.
        """
        self.rotation_matrix = np.array(new_rotation_matrix)

    def update_trajectory(self, new_center):
        """
        Update only the trajectory of the object.

        Parameters:
        - new_center (list or tuple): The new center coordinates to be added to the trajectory.
        """
        self.trajectory.append([new_center[0], new_center[1], new_center[2]])
    
    
    def get_id(self):
        """
        Get the unique identifier of the object.

        Returns:
        - int: The unique identifier (ID) of the object.
        """
        return self.id

    def get_rotation_matrix(self):
        """
        Get the rotation matrix of the object.

        Returns:
        - list or np.array: The rotation matrix representing the orientation of the object.
        """
        return self.rotation_matrix

    def get_center(self):
        """
        Get the center coordinates of the object.

        Returns:
        - list or tuple: The center coordinates (x, y, z) of the object.
        """
        return self.center

    def get_trajectory(self):
        """
        Get the trajectory of the object.

        Returns:
        - list of lists: The trajectory of the object, represented as a list of coordinates.
        """
        return self.trajectory

    def get_extent(self):
        """
        Get the extent or size of the object.

        Returns:
        - list or tuple: The extent or size of the object.
        """
        return self.extent
    
    def get_robot(self):
        return self.robot

    def __str__(self):
        """
        Return a string representation of the object.

        Returns:
        - str: A string containing information about the object, including its ID, center, extent,
               rotation matrix, and trajectory.
        """
        info_str = (
            f"Object ID: {self.id}\n"
            f"Center: {self.center}\n"
            f"Extent: {self.extent}\n"
            f"Rotation Matrix: {self.rotation_matrix}\n"
            f"Trajectory: {self.trajectory}\n"
        )
        return info_str


def first_time(object, checkpoints, v_max, delta_t, beta):
    """
    Initializes the orientation of the robot based on the first checkpoint.

    Parameters:
    object( Class Object ): Representation of a detected Object
    checkpoints (np.array): Equally spaced checkpoints of the path.
    v_max (float): The maximum velocity that the robot can have.
    delta_t (float): The time step.
    beta (float): Constant used to calculate the adjustment angle for each iteration.

    Returns:
    Returns:
    v (float): value of the velocity in the next iteration.
    teta_ajust (float): The angle (in degrees) that the robot is rotating in the next iteration.
    """

    center = object.get_center()
    rot_mat = object.get_rotation_matrix()

    # Find the closest checkpoint to the center of the bounding box.
    next_checkpoint = m.find_closest_checkpoint_new(center, checkpoints)

    # Calculate the front vector from the center to the next checkpoint.
    vector_front = m.create_vector(next_checkpoint, center)[:2]
    # Add a zero at the end to represent the third dimension.
    vector_front = np.append(vector_front, 0)

    # The rot_mat of the bbox may need to be updated based on the vector_front.
    updated_rot_mat = m.change_mat_bbox(rot_mat, vector_front)
    object.update_rotation_matrix(updated_rot_mat)

    # Calculate the angle between the x-axis of the bbox and the x-axis in inertial coordinates.
    teta_p = m.matrix_to_angle(updated_rot_mat)
    teta_p = m.normalize_angle(teta_p)

    # Calculate the angle between the vector from the center to the next checkpoint and the x-axis in inertial coordinates.
    teta_t = m.points_to_angle(center, next_checkpoint)
    teta_t = m.normalize_angle(teta_t)

    # Calculate the orientation error angle that the robot needs to rotate to face the first checkpoint.
    teta_e = teta_t - teta_p
    teta_ajust=d.ajust_angle(beta,teta_e,delta_t)
    #Calculation of the velocity value based on the teta_e value.
    v=m.velocity_value(teta_e,v_max)
    return v, teta_ajust

def other_time(object,checkpoints,v_max,delta_t,beta):
  rot_mat=object.get_rotation_matrix()
  center=object.get_center()
  """
  Given the ajustment angle and the velocity that the car moves returns the (x,y,z) of the exact position in the next time step. 
  Parameters:
    object( Class Object ): Representation of a detected Object.
        The detected object is a robot and we are giving instructions.
            The instructions are based on the detection (in the detected object (position, orientation...)).
    checkpoints (np.array): Equally spaced checkpoints of the path.
    v_max (float): The maximum velocity that the robot can have.
    delta_t (float): The time step.
    beta (float): Constant used to calculate the adjustment angle for each iteration.

  Returns:
    v (float): value of the velocity in the next iteration.
    teta_ajust (float): The angle (in degrees) that the robot is rotating in the next iteration.
  """
  teta_e=get_teta_e(object,checkpoints)
  #Calculate the angles that the robot is rotating in the next iteration.
  teta_ajust=d.ajust_angle(beta,teta_e,delta_t)
  v=m.velocity_value(teta_e,v_max)
  return v,teta_ajust

def get_teta_e(object,checkpoints):
    center=object.get_center()
    rot_mat=object.get_rotation_matrix()
    vector_front=object.get_vector_front()

    """
    Given, the checkpoints and the rot mat from the bbox, returns the teta_e.
    Parameters:
    object( Class Object ): Representation of a detected Object.
    checkpoints (np.array): Equally spaced checkpoints of the path.

    Returns:
    teta_e (np.array): The angle (in degrees) that the robot needs to rotate to face the next checkpoint (t).
    """
    #vector_front=m.create_vector(trajectory[-1],trajectory[-2])
    next_checkpoint=m.find_closest_checkpoint_new(center,checkpoints)
    #print("next_checkpoint",next_checkpoint)
    
    #The rot_mat of the bbox may need to be updated based on the vector_front.
    updated_rot_mat=m.change_mat_bbox(rot_mat,vector_front)
    object.update_rotation_matrix(updated_rot_mat)
    #The angle between the x_axis of the bbox and the x_inertial. 
    teta_p=m.matrix_to_angle(updated_rot_mat)
    #print("teta_p",teta_p)
    #The angle betwen the center_to_next_checkpoint and the x_inertial.
    teta_t=m.points_to_angle(center,next_checkpoint)
    #print("teta_t",teta_t)
    
    #The rotation that the robot needs to perform to be facing the next_checkpoint.
    teta_e=teta_t-teta_p
    return teta_e



class Driving():
    def __init__(self):
        """
        Variáveis que queremos armazenar:
            self.dic() é um dicionário cujas keys é o id do robot ao qual queremos dar instruções
                                            values são as instruções de condução(v,teta_ajust)
            self.control_path() é um dicionário que é suposto controlar se algum path foi alterado, se sim de qual robot e quais são os 
                                pontos que foram usados como checkpoints para abrir novo caminho.
        """

        self.actual_objects=[]
        self.dic={}

    
        self.v_max=0.4
        self.delta_t=0.5 #Tem de ser alterada para ser concordante com o timestamp das iterações
        self.beta=1 #Calc of the next_pos
        self.dist=0.3
        self.space_check=10
        self.space_check_strict=8
        self.spacing=0.1 #esta é a distancia a que estão os pontos num caminho.
        
    def incoming_box(self,bbox,robot_id,path):
        (center,extent,rotation_matrix)=bbox

        # Check if there is an object with the given robot_id
        matching_objects = [obj for obj in self.actual_objects if robot_id == obj.get_id()]
        
        if matching_objects:
        # Update the existing object
            obj=matching_objects[0]
            obj.update_object(center, extent, rotation_matrix)
            traj = obj.get_trajectory()
            vector_f = m.create_vector(traj[-1], traj[-2])
            obj.update_vector_front(vector_f)

            checkpoints= m.create_checkpoints_with_variable_spacing(path,spacing_away=self.space_check,spacing_near=self.space_check_strict)
            v,teta_ajust=other_time(obj,checkpoints,self.v_max,self.delta_t,self.beta)
            self.dic[robot_id]=(v,teta_ajust)


        else:
            # No object with the given robot_id, create a new one
            new_object = Object(robot_id, center, extent, rotation_matrix)
            self.actual_objects.append(new_object)

            checkpoints= m.create_checkpoints_with_variable_spacing(path,spacing_away=self.space_check,spacing_near=self.space_check_strict)
            v,teta_ajust=first_time(new_object,checkpoints,self.v_max,self.delta_t,self.beta)
            self.dic[robot_id]=(v,teta_ajust)











