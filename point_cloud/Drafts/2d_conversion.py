import open3d as o3d
import pyautogui

# Load a PointCloud from a .pcd file
point_cloud = o3d.io.read_point_cloud("Map1.pcd")

# Visualize the loaded PointCloud
o3d.visualization.draw_geometries([point_cloud])

# Capture the screen
screenshot = pyautogui.screenshot()

# Save the screenshot as a PNG file
screenshot.save("output_image.png")



"""
visualizer = o3d.visualization.Visualizer()
visualizer.create_window()
visualizer.get_render_option().point_size = 2  # Adjust point size as needed
visualizer.get_render_option().show_coordinate_frame = True  # Show coordinate frame
visualizer.get_render_option().background_color = [1, 1, 1]  # Set background color to white
visualizer.get_render_option().light_on = True  # Turn on lighting

# Add your point cloud to the visualizer
visualizer.add_geometry(point_cloud)

# Capture the screen image and save it
visualizer.capture_screen_image("output_image.png")"""