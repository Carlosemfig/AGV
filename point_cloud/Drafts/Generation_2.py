import open3d as o3d
import numpy as np

# To create the floor grid
width = 5  # Width of the floor
length = 7  # Length of the floor
spacing = 0.1  # Spacing between points
wall_height=2

# Create a grid of points to represent the floor
x = np.arange(0, width, spacing)
y = np.arange(0, length, spacing)
x, y = np.meshgrid(x, y)
z = np.zeros_like(x)  # Set the Z coordinate to zero for the floor
# Flatten the grid to create a point cloud
x = x.flatten()
y = y.flatten()
z = z.flatten()

print("este é o valor do ultimo",x[-1])



# Create a grid of points to represent the left wall
y_wall_l = np.arange(0, length, spacing)
z_wall_l= np.arange(0, wall_height, spacing)
y_wall_l, z_wall_l = np.meshgrid(y_wall_l, z_wall_l)
x_wall_l = np.zeros_like(y_wall_l)  # Set the Z coordinate to zero for the floor


# Create a grid of points to represent the right wall

y_wall_r=y_wall_l
z_wall_r=z_wall_l
x_wall_r=np.zeros_like(y_wall_r)
x_wall_r[x_wall_r == 0] = 4.9

# Flatten the grid to create a point cloud
x_wall_l = x_wall_l.flatten()
y_wall_l = y_wall_l.flatten()
z_wall_l = z_wall_l.flatten()

x_wall_r = x_wall_r.flatten()
y_wall_r = y_wall_r.flatten()
z_wall_r = z_wall_r.flatten()



x_global = np.concatenate((x, x_wall_l, x_wall_r))
y_global = np.concatenate((y, y_wall_l, y_wall_l))
z_global = np.concatenate((z, z_wall_l, z_wall_l))

#print(x_global.shape)
# Create a point cloud object and populate it with the structured data
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(np.vstack((x_global, y_global, z_global)).T)

# Visualize the structured point cloud
o3d.visualization.draw_geometries([point_cloud])
