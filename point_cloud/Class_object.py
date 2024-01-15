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

    def is_robot(self):
        self.robot=True
    
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
            print(f"Object ID: {id}, Is a Robot: {is_robot},Center: ({center}), Trajectory: ({trajectory})")

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