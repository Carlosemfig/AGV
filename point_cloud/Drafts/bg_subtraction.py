import open3d as o3d

# Load the background point cloud
bg = o3d.io.read_point_cloud("bg.pcd")

# Load the object point cloud
object = o3d.io.read_point_cloud("object_2.pcd")

def subtract_bg(bg_cloud,object_cloud):
  

    # Get the numpy arrays of points from the point clouds
    bg_points = bg_cloud.points
    object_points = object_cloud.points

    # Perform background subtraction by removing points in the object cloud from the background cloud
    result_points = [point for point in object_points if point not in bg_points]

    # Check if there are valid result points
    if len(result_points) == 0:
        print("No valid points left after subtraction.")
        return

    # Create a new point cloud from the result
    result_cloud = o3d.geometry.PointCloud()
    result_cloud.points = o3d.utility.Vector3dVector(result_points)

    # Save the result to a PCD file
    o3d.io.write_point_cloud("result.pcd", result_cloud)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(result_cloud)
    vis.run()
    vis.destroy_window()
    # Visualize the structured point cloud
    #o3d.visualization.draw_geometries([result_cloud])

if __name__ == "__main__":
    subtract_bg(bg,object)
