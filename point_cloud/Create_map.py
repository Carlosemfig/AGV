import MAIN as m
import numpy as np


space=0.1

#Definition of the lines that form the map, that will latter be walls
l1=m.create_straight_line((0,0),(0,8),space)
l2=m.create_straight_line((4,0),(4,4),space)
l3=m.create_straight_line((0,8),(4,8),space)
l4=m.create_straight_line((8,8),(16,8),space)
l5=m.create_straight_line((4,4),(24,4),space)
l6=m.create_straight_line((20,8),(20,16),space)
l7=m.create_straight_line((24,4),(24,12),space)
l8=m.create_straight_line((24,12),(28,12),space)
l9=m.create_straight_line((24,14),(28,14),space)
l10=m.create_straight_line((24,16),(28,16),space)

final_line=m.merge_arrays(l1,l2,l3,l4,l5,l6,l7,l8,l9,l10)


#Converts the array of the combination of the lines into walls
def create_wall(line,height,spacing):
    """
    Creates a wall along a line in 3D space.

    Parameters:
    line (np array): Representation of the line in a np array format (shape: (n, 3)).
    height (float): Height of the wall.
    spacing (float): Spacing between layers of the wall.

    Returns:
    np array: 3D coordinates of the wall points (shape: (m, 3)).
    """
    z_values = np.arange(0, height, spacing)
    wall_points = []

    for z in z_values:
        points = np.array([line[:, 0], line[:, 1], np.full_like(line[:, 0], z)]).T
        wall_points.extend(points)

    return np.array(wall_points)

Bg=create_wall(final_line,4,space)
#Creation of the Parts
obj1=m.create_cubic_object((2,1),0.3,0.3,0.1)
obj2=m.create_cubic_object((1,0.5),0.3,0.3,0.1)
obj3=m.create_cubic_object((3,0.5),0.3,0.3,0.1)
obj4=m.create_cubic_object((1,1.5),0.3,0.3,0.1)
obj5=m.create_cubic_object((3,1.5),0.3,0.3,0.1)
Parts=m.merge_arrays(obj1,obj2,obj3,obj4,obj5)

#This is the np.array format of the background that will be imported
#Contains the walls and the Parts

Background=m.merge_arrays(Bg,Parts)



"""___________________________________________________________________________"""

#From the departure to the parts
d1=m.create_straight_line((26,15),(22,15),space)
d2=m.create_straight_line((22,15),(22,6),space)
d3=m.create_straight_line((22,6),(2,6),space)
d4=m.create_straight_line((2,6),(2,2),space)
d11=m.create_straight_line((26,13),(22,13),space)
d12=m.create_straight_line((22,13),(22,6),space)

#This are the paths that can be imported
D1_parts=m.merge_arrays(d1,d2,d3,d4)
D2_parts=m.merge_arrays(d11,d12,d3,d4)


#From the Parts to the tasks
t1=m.create_straight_line((2,2),(2,6),space)
t2=m.create_straight_line((2,6),(6,6),space)
t3=m.create_straight_line((6,6),(6,10),space)
t21=m.create_straight_line((2,6),(18,6),space)
t22=m.create_straight_line((18,6),(18,10),space)

#This are the paths that can be imported
Parts_T1=m.merge_arrays(t1,t2,t3)
Parts_T2=m.merge_arrays(t1,t21,t22)

