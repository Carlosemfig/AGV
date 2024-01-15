import open3d as o3d
import numpy as np

# To create the floor grid
width = 5  # Width of the floor
length = 7  # Length of the floor
spacing = 0.1  # Spacing between points
wall_height=2
plane=0

length=[0,7]
width=[0,5]
height=[0,2]

def constant_z(width, length, plane, spacing):
    # Create a grid of points to represent the floor
    x = np.arange(width[0], width[1], spacing)
    y = np.arange(length[0], length[1], spacing)
    x, y = np.meshgrid(x, y)
    z = np.zeros_like(x)  # Set the Z coordinate to zero for the floor
    if plane != 0:
        z[z == 0] = plane

    
    # Flatten the grid to create a point cloud
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    return x, y, z


def constant_y(width, height, plane, spacing):
    # Create a grid of points to represent the floor
    x = np.arange(width[0], width[1], spacing)
    z= np.arange(height[0], height[1], spacing)
    x, z = np.meshgrid(x, z)
    y = np.zeros_like(x)  # Set the Z coordinate to zero for the floor
    if plane != 0:
        y[y == 0] = plane

    
    # Flatten the grid to create a point cloud
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    return x, y, z


def constant_x(length, height, plane, spacing):
    # Create a grid of points to represent the floor
    y = np.arange(length[0], length[1], spacing)
    z= np.arange(height[0], height[1], spacing)
    y, z = np.meshgrid(y, z)
    x = np.zeros_like(y)  # Set the Z coordinate to zero for the floor
    if plane != 0:
        x[x == 0] = plane

    
    # Flatten the grid to create a point cloud
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    return x, y, z


def create_cubic_object(center, width, length, height, spacing):
    # Calculate the dimensions of the cubic object
    h_width= width/2
    h_lenght=length/2


    width_range = np.arange(center[0] - h_width, center[0] + h_width, spacing)
    length_range = np.arange(center[0] - h_lenght, center[0] + h_lenght, spacing)
    height_range = np.arange(0, height, spacing)

    # Create a grid of points to represent the cubic object
    x, y, z = np.meshgrid(width_range, length_range, height_range, indexing='ij')

    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    return x, y, z

# Example usage of the function
center_cub=(1,1)
width_cub = 0.6  # Width range
length_cub = 0.6  # Length range
height_cub = 0.6  # Height of the cubic object
spacing=0.1

x_cub, y_cub, z_cub = create_cubic_object(center_cub,width_cub, length_cub, height_cub, spacing)

x_cub2, y_cub2, z_cub2 = create_cubic_object((4,6),width_cub, length_cub, height_cub, spacing)



x, y, z = constant_z(width, length, 0, spacing)

x_wall_l, y_wall_l, z_wall_l=constant_x(length, height, 0, spacing)

x_wall_r, y_wall_r, z_wall_r = constant_x(length, height, 4.9, spacing)

x_wall_1, y_wall_1, z_wall_1 = constant_y(width,height, 6.9, spacing)

length_mid=[0,5]

x_wall_m1, y_wall_m1, z_wall_m1=constant_x(length_mid, height, 1.9, spacing)
x_wall_m2, y_wall_m2, z_wall_m2=constant_x(length_mid, height, 2.9, spacing)

width_mid= [2,3]

x_wall_2, y_wall_2, z_wall_2 = constant_y(width_mid,height, 5, spacing)




x_global = np.concatenate(( x_wall_l, x_wall_r,x_wall_1, x_wall_m1,x_wall_m2,x_wall_2,x_cub,x_cub2))
y_global = np.concatenate(( y_wall_l, y_wall_r,y_wall_1, y_wall_m1,y_wall_m2,y_wall_2,y_cub,y_cub2))
z_global = np.concatenate(( z_wall_l, z_wall_r,z_wall_1, z_wall_m1,z_wall_m2,z_wall_2,z_cub,z_cub2))

#print(x_global.shape)
# Create a point cloud object and populate it with the structured data
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(np.vstack((x_global, y_global, z_global)).T)



o3d.io.write_point_cloud("object_2.pcd", point_cloud)

# Visualize the structured point cloud
o3d.visualization.draw_geometries([point_cloud])

# Save the image as a PNG or JPEG 

o3d.visualization.draw_geometries([point_cloud]).capture_screen_image("Map1.png", True)

