import open3d as o3d
from sklearn.cluster import DBSCAN
import numpy as np

def main():
    # Load the point cloud
    cloud = o3d.io.read_point_cloud("result.pcd")

    # Convert the point cloud to a NumPy array for clustering
    points = np.asarray(cloud.points)

    # Perform DBSCAN clustering
    eps = 0.2  # Adjust the epsilon (maximum distance between points in the same cluster) as needed
    min_samples = 10  # Adjust the minimum number of points required to form a cluster as needed
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)

    # Get cluster labels from DBSCAN
    cluster_labels = clustering.labels_

    print("o que é clustering.labels_", cluster_labels)

    # Determine the number of clusters (objects)
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"Number of objects identified: {num_clusters}")

    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Iterate through the clusters and represent each object
    for cluster_id in range(num_clusters):
        object_points = points[cluster_labels == cluster_id]

        # Calculate the centroid of the object
        centroid = np.mean(object_points, axis=0)
        print(f"Object {cluster_id} - Centroid: {centroid}")


        # Calculate the bounding box of the object
        min_bound = np.min(object_points, axis=0)
        max_bound = np.max(object_points, axis=0)

        # Create a colored point cloud for the object's centroid
        centroid_color = [0, 0, 1]  # Blue color
        centroid_cloud = o3d.geometry.PointCloud()
        centroid_cloud.points = o3d.utility.Vector3dVector([centroid])
        centroid_cloud.colors = o3d.utility.Vector3dVector([centroid_color])

        # Create a colored bounding box for the object
        object_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(object_points))
        object_bbox.color = [0, 1, 0]  # Green color

        # Add the objects to the visualizer
        vis.add_geometry(centroid_cloud)
        vis.add_geometry(object_bbox)

        
    # Add the entire point cloud for context
    vis.add_geometry(cloud)

    # Run the visualizer
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
