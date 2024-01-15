import numpy as np
import open3d as o3d
import os

# Helper parameters class containing variables that will change in the callback function
class params():
    # voxels counter that will stop the voxel mesh generation when there are no more voxels in the voxel grid
    counter = 0
    vox_mesh=o3d.geometry.TriangleMesh()

# Voxel builder callback function
def build_voxels(vis):
    # check if the counter is more than the amount of voxels
    if params.counter < len(voxels_all):
        # get the size of the voxels
        voxel_size = voxel_grid.voxel_size
        # create a box primitive of size 1x1x1
        cube=o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
        # paint the box uniformly with the color of the voxel
        cube.paint_uniform_color(voxels_all[params.counter].color)
        # scale the box to the size of the voxel
        cube.scale(voxel_size, center=cube.get_center())
        # get the center position of the current voxel
        voxel_center = voxel_grid.get_voxel_center_coordinate(voxels_all[params.counter].grid_index)
        # translate the box to the voxel center
        cube.translate(voxel_center, relative=False)
        # add the box primitive to the voxel mesh
        params.vox_mesh+=cube
        
        # on the first loop create the geometry and on subsequent iterations update the geometry
        if params.counter==0:
            vis.add_geometry(params.vox_mesh)
        else:
            vis.update_geometry(params.vox_mesh)

        # update the renderer
        vis.update_renderer()
        # tick up the counter
        params.counter+=1
    
def rotate_view(vis):
    ctr = vis.get_view_control()
    ctr.rotate(10.0, 0.0)
    return False



   
# Callback function used to construct and rotate the voxel meshes
def rotate_and_change(vis):

    # When the counter is 0 generate the voxel grid and construct the voxel mesh
    if params.counter == 0:
        # generate the voxel grid using the voxel sizes setup in the params class
        voxel_grid=o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=params.voxel_sizes[params.sizes_counter])
        # get all voxels in the voxel grid
        voxels_all= voxel_grid.get_voxels()
        # geth the calculated size of a voxel
        voxel_size = voxel_grid.voxel_size
        # loop through all the voxels
        for voxel in voxels_all:
            # create a cube mesh with a size 1x1x1
            cube=o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
            # paint it with the color of the current voxel
            cube.paint_uniform_color(voxel.color)
            # scale the box using the size of the voxel
            cube.scale(voxel_size, center=cube.get_center())
            # get the center of the current voxel
            voxel_center = voxel_grid.get_voxel_center_coordinate(voxel.grid_index)
            # translate the box to the center of the voxel
            cube.translate(voxel_center, relative=False)
            # add the box to the TriangleMesh object
            params.vox_mesh+=cube
        
        # on the first run of the callback loop initialize the Triangle mesh by adding it to the Visualizer. In subsequent iterations just update the geometry
        if params.initialize:
            vis.add_geometry(params.vox_mesh)
            params.initialize = False
        else:
            vis.update_geometry(params.vox_mesh)


    # We create a 3D rotation matrix from x,y,z rotations, the rotations need to be given in radians
    R = params.vox_mesh.get_rotation_matrix_from_xyz((0, np.deg2rad(2), 0))
    # The rotation matrix is applied to the specified object - in our case the voxel mesh. We can also specify the rotation pivot center
    params.vox_mesh.rotate(R, center=(0, 0, 0))
    # tick the counter up
    params.counter+=1
    # For the changes to be seen we need to update both the geometry that has been changed and to update the whole renderer connected to the visualizer
    vis.update_geometry(params.vox_mesh)
    
    vis.update_renderer()

    # When the counter gets to 180 we zero it out. This is done because we rotate the mesh by 2 degrees on an iteration
    if params.counter >= 180:
        params.counter=0
        # we tick the voxel size counter
        params.sizes_counter +=1
        # if the voxel size counter becomes equal to the size of the voxel sizes array, reset it
        if params.sizes_counter >= len(params.voxel_sizes):
            params.sizes_counter=0
        # each time we clear the mesh. This is important, because without it we will just add more and more box primitives to the mesh object
        params.vox_mesh.clear()
  

import laspy
import numpy as np
# Path to the LAS file
las_file_path = r"C:\Users\hvendas\Desktop\point_cloud to mesh\python-3d-analysis-libraries\How to Voxelize Meshes and Point Clouds in Python\point_cloud\Baltimore.las"

# Open the LAS file
las_file = laspy.file.File(las_file_path, mode="r")
# Get the LAS point format
point_format = las_file.point_format

# Print the available attribute names
attribute_names = point_format.lookup.keys()

print("Available attribute names:", attribute_names)
#['X', 'Y', 'Z', 'gps_time']

#este dá os pontos x,y,z caracteristicos que permitem transformar numa grid
points = np.column_stack((las_file.x, las_file.y, las_file.z))

gps_time_array = np.array(las_file.gps_time)
print(gps_time_array)


"""
from sklearn.cluster import DBSCAN

# Assuming you have 'gps_time_array'
dbscan = DBSCAN(eps=100, min_samples=1)
labels = dbscan.fit_predict(gps_time_array.reshape(-1, 1))
#não dá para usar as cores, porque não tem atributo cores
#colors = np.column_stack((las_file.red, las_file.green, las_file.blue))
"""




# Initialize a point cloud object
#pcd = o3d.geometry.PointCloud()

# Add the points and colors as Vectors
# pcd.points = o3d.utility.Vector3dVector(points)


pcd = o3d.io.read_point_cloud(r"C:\Users\inesolopes\Documents\CARLA_0.9.14\WindowsNoEditor\PythonAPI\util\tutorial\sensor_1_output\001347.ply")

import open3d as o3d

# Assuming you have already defined your voxel grid

# Create a visualizer object
vis = o3d.visualization.Visualizer()

# Create a window, name it, and set its dimensions
vis.create_window(window_name='Voxel Grid Visualization', width=800, height=600)

voxel_grid=o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=0.5)
# Get all the voxels in the voxel grid
voxels_all= voxel_grid.get_voxels()

# Add the voxel grid mesh to the visualizer
vis.add_geometry(voxel_grid)

# Run the visualizer
vis.run()

# Once the visualizer is closed, destroy the window and clean up
vis.destroy_window()





