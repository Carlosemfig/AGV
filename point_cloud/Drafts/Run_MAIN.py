import open3d as o3d
import numpy as np
import MAIN as m



bg=m.load_pc("bg.pcd")
bg=m.pc_to_array(bg)

# Create a path
line1_start = (1.0, 0.0)
line1_end = (1.0, 6.0)

line2_start = (1.0, 6.0)
line2_end = (4.0, 6.0)

line3_start = (4.0, 6.0)
line3_end = (4.0, 0.0)

space=0.1

path1 =m.create_straight_line(line1_start, line1_end, space)
path2=m.create_straight_line(line2_start,line2_end,space)
path3=m.create_straight_line(line3_start,line3_end,space)

global_path= m.merge_arrays(path1,path2)
global_path=m.merge_arrays(global_path,path3)


#create an object
center_cub=(1,1)
width_cub = 0.6  # Width range
length_cub = 0.6  # Length range
height_cub = 0.6  # Height of the cubic object



cube_1=m.create_cubic_object(m.random_coordinate_from_array(global_path),width_cub,length_cub,space)
cube_2=m.create_cubic_object(m.random_coordinate_from_array(global_path),width_cub,length_cub,space)


final_map=m.merge_arrays(global_path,bg)


object_1=m.merge_arrays(final_map,cube_1)
point_cloud1=m.array_to_pc(object_1)
#m.visualize(point_cloud1)

object_2=m.merge_arrays(final_map,cube_2)
point_cloud2=m.array_to_pc(object_2)
#m.visualize(point_cloud2)

list=[point_cloud1,point_cloud2]


import open3d as o3d
import time



# Create a visualization window
visualizer = o3d.visualization.Visualizer()
visualizer.create_window()

visualizer.add_geometry(point_cloud1)
visualizer.update_geometry(point_cloud1)
visualizer.poll_events()
visualizer.update_renderer()
#o3d.utility.wait(2000)  # Display the first point cloud for 2 seconds
time.sleep(2)
# Remove the first point cloud
visualizer.clear_geometries()

# Add the second point cloud
visualizer.add_geometry(point_cloud2)
visualizer.update_geometry(point_cloud2)
visualizer.poll_events()
visualizer.update_renderer()
time.sleep(2)
# Keep the visualization window open
print("Press Q or close the window to exit.")
visualizer.run()  # This keeps the window open

# Close the visualization window
visualizer.destroy_window()