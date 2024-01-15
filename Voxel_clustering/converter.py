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

#print("Available attribute names:", attribute_names)
#['X', 'Y', 'Z', 'intensity', 'flag_byte', 'raw_classification', 'scan_angle_rank', 'user_data', 'pt_src_id', 'gps_time']

#este dá os pontos x,y,z caracteristicos que permitem transformar numa grid
points = np.column_stack((las_file.x, las_file.y, las_file.z))

#não dá para usar as cores, porque não tem atributo cores
#colors = np.column_stack((las_file.red, las_file.green, las_file.blue))

# Use 'intensity' as a proxy for colors
colors = np.column_stack((las_file.intensity, las_file.intensity, las_file.intensity))
