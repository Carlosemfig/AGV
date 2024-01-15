
import numpy as np
import math

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



class EuclideanDistTracker3D:
    """
    This class implements a 3D object tracker based on Euclidean distance.
    It assigns unique IDs to objects and updates their positions over time.
    
    Attributes:
    - stored_objects (list): List to store the center positions of the objects.
    - id_count (int): Counter for assigning unique IDs to objects.
    """
    def __init__(self):
        """
        Initialize the EuclideanDistTracker3D object tracker.

        Attributes:
        - stored_objects (list): List to store the center positions of the objects.
        - id_count (int): Counter for assigning unique IDs to objects.
        """
        self.stored_objects = []
        self.id_count=0

    def update(self, objects_dict, distancia):
        """
        Update the object tracker with the latest detections.

        Parameters:
        - objects_dict (dict): A dictionary containing all the bounding boxes (bboxes) detected in a time step.
                              The keys are object IDs, and the values are tuples (center, extent, rotation_matrix).
        - distancia (float): The maximum distance for associating a detection with a stored object.

        Returns:
        - None
        """
        iteration=[]
        #Itera a lista de objectos detetados
        for obj_id, (center, extent, rotation_matrix) in objects_dict.items():
            # Find out if that object was detected already
            same_object_detected = False
            #Itera sobre todos os objetos que temos armazenados
            for obj in self.stored_objects:
                #Verifica a distancia do centro dos objectos detetados com os armazenados
                dist = np.linalg.norm(center - obj.get_center())
                if dist < distancia:  # Adjust the threshold as needed´
                    #Neste caso o objecto detetado é um dos que já estavam armazenados, 
                    #Por isso só queremos atualizar o objecto armazenado
                    obj.update_object(center, extent, rotation_matrix)
                    
                    traj=obj.get_trajectory()
                    vector_f=m.create_vector(traj[-1],traj[-2])
                    obj.update_vector_front(vector_f)

                    iteration.append(obj.get_id())
                    same_object_detected = True
                    break

            # New object is detected; assign the ID to that object
            if not same_object_detected:
                new_object = Object(self.id_count, center, extent, rotation_matrix)
                iteration.append(new_object.get_id())
                self.id_count=self.id_count + 1
                self.stored_objects.append(new_object)

        # Remove objects from the tracker if they are not in the new detection
        self.stored_objects = [obj for obj in self.stored_objects if obj.get_id() in iteration]

    def get_stored_objects(self):
        """
        Get the list of stored objects.

        Returns:
        - list: List containing objects with their center positions, extents, rotation matrices, and trajectories.
        """
        return self.stored_objects
    
    def print_stored_objects(self):
        """
        Print information about the stored objects.

        Returns:
        - None
        """
        print("Stored Objects:")
        print_dic={}
        for obj in self.stored_objects:
            id=obj.get_id()
            center=obj.get_center()
            is_robot=obj.get_robot()
            #rot_mat=obj.get_rotation_matrixt()
            trajectory=obj.get_trajectory()
            vector_front=obj.get_vector_front()
            print(f"Object ID: {id}, Is a Robot: {is_robot},Center: ({center}), Trajectory: ({trajectory}), Vector Front: ({vector_front})")

    def get_ids(self):
        """
        Get the IDs of the stored objects.

        Returns:
        - list: List containing the unique identifiers (IDs) of the stored objects.
        """
        ids=[]
        ids = [obj.get_id() for obj in self.stored_objects]
        return ids
    
    def __iter__(self):
        """
        Make the class iterable by returning an iterator over stored_objects.

        Returns:
        - iterator: Iterator over stored_objects.
        """
        return iter(self.stored_objects)
    
import MAIN as m
import MAIN_DRIVING as d


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


class Robot_atribuition:
    """
    Está feito para associar o id de um robot, o caminho que ele é suposto 
    fazer e o id do objecto detetado a que corresponde o robot.

    """
    def __init__(self):
        """
        self.identification: Key: é o número de identificação do robot
                            Values: path que aquele robot é suposto fazer.
                                    object id objecto detetado que corresponde ao robot.
        """
        self.identification={}
    
    def attribute_path(self, robot_id, path):
        """
        Atribuição do caminho a um id de um robot
        """
        # Check if the robot_id exists in the dictionary
        if robot_id in self.identification:
            current_path, current_object_id = self.identification[robot_id]
            self.identification[robot_id] = (path, current_object_id)
        else:
            # If the robot_id doesn't exist, add a new entry with the given path
            self.identification[robot_id] = (path, None)

    def attribute_object_id(self, robot_id, object_id):
        """
        Attribution of the object ID to a robot_id.
        """
        # Check if the robot_id exists in the dictionary
        if robot_id in self.identification:
            current_path, current_object_id = self.identification[robot_id]
            self.identification[robot_id] = (current_path, object_id)
        else:
            # If the robot_id doesn't exist, add a new entry with the given object_id
            self.identification[robot_id] = (None, object_id)
    
    def get_object_id(self, robot_id):
        """
        Get the object ID associated with a robot ID.
        """
        if robot_id in self.identification:
            _, object_id = self.identification[robot_id]
            return object_id
        else:
            return None
        
    def get_path(self, robot_id):
        """
        Get the object ID associated with a robot ID.
        """
        if robot_id in self.identification:
            path,_ = self.identification[robot_id]
            return path
        else:
            return None

    def get_identification(self):
        return self.identification()
    
class Driving:
    """
    Compilação das funções de condução. Inicialização é composta por duas partes:
        1 - Variáveis que queremos armazenar.
        2 - Constantes que precisam de ser inicializadas para realizar os cálculos. 
    """
    def __init__(self):
        """
        Variáveis que queremos armazenar:
            self.dic() é um dicionário cujas keys é o id do robot ao qual queremos dar instruções
                                            values são as instruções de condução(v,teta_ajust)
            self.control_path() é um dicionário que é suposto controlar se algum path foi alterado, se sim de qual robot e quais são os 
                                pontos que foram usados como checkpoints para abrir novo caminho.
        """
        self.dic={}
        self.control_path={}
    
        self.v_max=0.4
        self.delta_t=0.5 #Tem de ser alterada para ser concordante com o timestamp das iterações
        self.beta=1 #Calc of the next_pos
        self.dist=0.3
        self.space_check=10
        self.space_check_strict=8
        self.spacing=0.1 #esta é a distancia a que estão os pontos num caminho.
        


    #o que fazer quando o objecto encontrado corresponde a um dos objectos
    def first_time(self,robot_id):
        """
        Updates the self.dic with the velocity and the angle for a robot, for the situation that is the first time the robot is working. 

        Parameter:
         - object( Class Object ) - The object that has being detected as the robot. Gives information regarding the actual state of the robot.
         - Path (np.array) - The path the robot is supost to do. (the next instruction will be based on where is the next checkpoint).

        """

        #If the input is robot id and not the object and the path
        
        object=Robot_atribuition.get_object_id(robot_id)
        path=Robot_atribuition.get_path(robot_id)
        
        checkpoints= m.create_checkpoints_with_variable_spacing(path,spacing_away=self.space_check,spacing_near=self.space_check_strict)
        v,teta_ajust=first_time(object,checkpoints,self.v_max,self.delta_t,self.beta)
        self.dic[object.get_id()]=(v,teta_ajust)
    
    def other_time(self,object,path):
        """
        Updates the self.dic with the velocity and the angle for a robot, for the situation that is the first time the robot is detected.

        Parameter:
         - object( Class Object ) - The object that has being detected as the robot. Gives information regarding the actual state of the robot.
         - Path (np.array) - The path the robot is supost to do. (the next instruction will be based on where is the next checkpoint).

        """
        #If the input is robot id and not the object and the path
        #object=Robot_atribuition.get_object_id(robot_id)
        #path=Robot_atribuition.get_path(robot_id)
        checkpoints= m.create_checkpoints_with_variable_spacing(path,spacing_away=self.space_check,spacing_near=self.space_check_strict)
        v,teta_ajust=other_time(object,checkpoints,self.v_max,self.delta_t,self.beta)
        self.dic[object.get_id()]=(v,teta_ajust)
    

    def get_dic(self):
        return self.dic
    
    def print_dic(self):

        print("Driving instructions:")
        for obj_id, (velocity, teta_ajust) in self.dic.items():
            print(f"Object ID: {obj_id} = (Velocity: {velocity}, Adjustment Angle: {teta_ajust})")
    
    #if the object detected is not an robot

    def is_obstacle_in_path(self,object,path):
        #returns a bolean if is in path or not
        return d.is_box_in_path(object,path)
    
    def update_path(self,obstacle, robot, path):
        """
        This is used for the situaation where the detected obstacle is still in the path.
        obstacle and robot are both objects, from the object class. 
        The obstacle is the detected objet that is in the path.
        The robot is the robot associated with that path."""
        
        checkpoints= m.create_checkpoints_with_variable_spacing(path,spacing_away=self.space_check,spacing_near=self.space_check_strict)
        close_check=d.find_closest_checkpoints(obstacle.get_center(),checkpoints,tresh=1)
        
        #self.control_path[robot_id]=(close_check)
        
        deviation_point=d.find_deviation_point(obstacle,close_check,robot,margin=0.5)
        path=d.update_path(close_check,deviation_point,path,self.spacing)
        return path
    
