import numpy as np
import plotly.graph_objects as go
import networkx as nx
from PIL import Image
import yaml
from scipy.spatial import Voronoi
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
from utils import *
from scipy.ndimage import distance_transform_edt, binary_dilation

def pgm2occupancy(pgm_file, occupied_thresh=0.65, free_thresh=0.196):
    img = Image.open(pgm_file)
    img = img.convert('L')  # 'L' mode is for grayscale
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]

    occupancy_grid = np.zeros_like(img_array, dtype=int)
    occupancy_grid[img_array > occupied_thresh] = 0
    occupancy_grid[img_array < free_thresh] = 1
    occupancy_grid[(img_array >= free_thresh) & (img_array <= occupied_thresh)] = -1

    return occupancy_grid

def plot_occupancy_grid(occupancy_grid):
    fig = go.Figure(data=go.Heatmap(
        z=occupancy_grid,
        colorscale='gray',
        reversescale=True,
        showscale=True
    ))
    fig.update_layout(
        title='Occupancy Grid',
        xaxis_title='X (cells)',
        yaxis_title='Y (cells)',
        yaxis=dict(autorange='reversed')
    )
    fig.show()

class patrol_graph():
    def __init__(self, yaml_path, pgm_file) -> None:
        with open(yaml_path, 'r') as stream:
            self.map_yaml = yaml.safe_load(stream)
        self.resolution = self.map_yaml['resolution']  # meters/pixel
        self.origin = self.map_yaml['origin']
        self.pgm_file = pgm_file

        self.occupancy_grid = pgm2occupancy(self.pgm_file)
        self.astar = AStar(self.occupancy_grid)  # Initialize A* pathfinding

    def gen_voronoi(self, distance_threshold=10, dilation_radius=1):
        # Use resolution to adjust distance threshold
        # distance_threshold = int(distance_threshold / self.resolution)
        
        # Step 1: Expand the obstacles using max pooling
        max_pool_size = 2 * dilation_radius + 1
        expanded_obstacles = np.zeros_like(self.occupancy_grid)
        
        # Perform max pooling
        for i in range(self.occupancy_grid.shape[0]):
            for j in range(self.occupancy_grid.shape[1]):
                # Define the window boundaries
                i_start = max(0, i - dilation_radius)
                i_end = min(self.occupancy_grid.shape[0], i + dilation_radius + 1)
                j_start = max(0, j - dilation_radius)
                j_end = min(self.occupancy_grid.shape[1], j + dilation_radius + 1)
                
                # Take maximum value in the window
                window = self.occupancy_grid[i_start:i_end, j_start:j_end]
                expanded_obstacles[i, j] = np.min(window)

        expanded_obstacles = expanded_obstacles.astype(int)  # Convert back to int (1 = obstacle, 0 = free space)
        # expanded_obstacles[np.where(self.occupancy_grid == 0)] = -1  # Mark unknown cells as -1   

        # Step 2: Find free spaces in the modified occupancy grid
        free_spaces = np.array(np.where(expanded_obstacles == 1)).T
        # free_spaces = np.array(np.where(self.occupancy_grid == 1)).T
        print(f"Total free spaces in modified grid: {free_spaces.shape[0]}")

        # If no valid spaces are found, exit
        if free_spaces.shape[0] == 0:
            print("No valid spaces found in the modified grid. Exiting...")
            return

        # Step 3: Select Voronoi centers based on the distance threshold
        centers = []
        for i in range(free_spaces.shape[0]):
            if len(centers) == 0:
                centers.append(free_spaces[i])
            else:
                # Check the minimum distance from current center to existing ones
                min_dist = np.min(np.linalg.norm(np.array(centers) - free_spaces[i], axis=1))
                if min_dist >= distance_threshold:
                    centers.append(free_spaces[i])

        # Convert centers to numpy array and ensure it's 2D
        centers = np.array(centers)

        # Check if centers is empty
        if centers.shape[0] == 0:
            print("No valid Voronoi centers found. Exiting...")
            return

        # Ensure centers have the correct 2D shape (n x 2)
        if centers.ndim == 1:
            centers = centers.reshape(-1, 2)

        # Step 4: Compute Voronoi diagram
        try:
            vor = Voronoi(centers)
        except ValueError as e:
            print(f"Error generating Voronoi diagram: {e}")
            return

        # Step 5: Plot the occupancy grid
        fig = go.Figure()

        # Add occupancy grid
        fig.add_trace(go.Heatmap(
            z=self.occupancy_grid,
            colorscale='gray',
            showscale=False
        ))

        # Add Voronoi centers as scatter points
        fig.add_trace(go.Scatter(
            x=centers[:, 1],
            y=centers[:, 0],
            mode='markers',
            marker=dict(color='red'),
            name='Voronoi centers'
        ))

        # Add text annotations for node IDs
        annotations = []
        for i, center in enumerate(centers):
            annotations.append(dict(
                x=center[1] + 5,
                y=center[0] + 5,
                text=str(i),
                showarrow=False,
                font=dict(color='blue', size=12)
            ))

        fig.update_layout(
            title='Voronoi Diagram with MST',
            annotations=annotations,
            yaxis=dict(autorange='reversed')
        )

        # Step 7: Construct and plot the MST (minimum spanning tree) for the Voronoi centers
        mst_edges, all_paths = self.plot_mst(centers)

        # Add the paths to the figure
        for path in all_paths:
            fig.add_trace(go.Scatter(
                x=path[:, 1],
                y=path[:, 0],
                mode='lines',
                line=dict(color='green', width=1),
                showlegend=False
            ))

        # Show the figure
        fig.show()

        # Step 8: Save the Graph in the required format
        generate_graph(centers=centers, mst_edges=mst_edges, filename=f"{self.pgm_file.split('.')[0]}.graph")

    def plot_mst(self, centers):
        # Compute pairwise distances between Voronoi centers
        dist_matrix = cdist(centers, centers, 'euclidean')

        # Use Kruskal's algorithm (via scipy minimum spanning tree function)
        mst = minimum_spanning_tree(dist_matrix)

        # Convert the sparse matrix to a dense matrix for easier processing
        mst = mst.toarray()

        # Initialize a list to store paths
        all_paths = []

        # Plot the MST by connecting the Voronoi centers with pathfinding checks
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                if mst[i, j] > 0:  # If there is an edge
                    start = tuple(centers[i])
                    end = tuple(centers[j])

                    # Use A* to check if a path exists
                    path = self.astar.a_star(start, end)

                    if path:
                        # If a valid path exists, collect the path
                        path = np.array(path)
                        all_paths.append(path)

        # Extract edges from the MST and calculate the distance
        mst_edges = []
        for i in range(mst.shape[0]):
            for j in range(i + 1, mst.shape[1]):
                if mst[i, j] > 0:  # If there's an edge between node i and node j
                    mst_edges.append((i, j, mst[i, j]))  # Add (node1, node2, distance)

        return mst_edges, all_paths

if __name__ == "__main__":
    G = patrol_graph('graph/map_name.yaml', 'graph/map_name.pgm')
    G.gen_voronoi(distance_threshold=100, dilation_radius=3)
