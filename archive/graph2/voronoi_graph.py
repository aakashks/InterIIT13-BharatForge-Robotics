import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image
import yaml
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
from utils import *
from scipy.ndimage import distance_transform_edt, binary_dilation
import numpy
from pgm_reader import Reader
import sys
import torch

numpy.set_printoptions(threshold=sys.maxsize)

def pgm2occupancy(pgm_file, occupied_thresh=0.65, free_thresh=0.196):
    img = Image.open(pgm_file)
    img = img.convert('L')  # 'L' mode is for grayscale
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    occupancy_grid = np.zeros_like(img_array, dtype=int)
    occupancy_grid[img_array > 0.9] = 0
    occupancy_grid[img_array <= 0.9] = 1
    # occupancy_grid[(img_array <= 0) & (img_array >= 0)] = 0
    # occupancy_grid[(img_array >= free_thresh) & (img_array <= occupied_thresh)] = -1
    # occupancy_grid[(img_array <= 0.9)] = -1


    # image = Reader().read_pgm(pgm_file)
    # img_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    # with open('file.txt', 'w') as file:
    #     for line in img_array:
    #         file.write(str(line) + '\n')
    # assert False
    return occupancy_grid

def plot_occupancy_grid(occupancy_grid):
    plt.imshow(occupancy_grid, cmap='gray', origin='lower')
    plt.colorbar(label='Occupancy')
    plt.title('Occupancy Grid')
    plt.xlabel('X (cells)')
    plt.ylabel('Y (cells)')
    plt.show()

def expand_obstacles(occupancy_grid, dilation_radius=20):
    # Perform binary dilation to expand obstacles
    input_tensor = torch.tensor(input_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Apply max pooling with the given pool size
    pooled_tensor = F.max_pool2d(input_tensor, kernel_size=pool_size, stride=pool_size)
    
    # Remove batch and channel dimensions to return the result as a 2D matrix
    return pooled_tensor.squeeze(0).squeeze(0).numpy()
    

class patrol_graph():
    def __init__(self, yaml_path, pgm_file) -> None:
        
        with open(yaml_path, 'r') as stream:
            self.map_yaml = yaml.safe_load(stream)
        self.pgm_file = pgm_file
        self.occupancy_grid = pgm2occupancy(self.pgm_file)

        self.resolution = self.map_yaml['resolution'] # meters/pixel
        self.origin = self.map_yaml['origin']
        self.height = self.occupancy_grid.shape[0]
        self.width = self.occupancy_grid.shape[1]
        
        self.astar = AStar(self.occupancy_grid)  # Initialize A* pathfinding
    
    def gen_voronoi(self, distance_threshold=40, dilation_radius=20):
    
        # Step 1: Expand obstacles in the occupancy grid
        self.expanded_grid = expand_obstacles(self.occupancy_grid, dilation_radius)
        # Step 2: Find free spaces in the modified occupancy grid
        free_spaces = np.array(np.where(self.occupancy_grid == 0)).T #NOTE: see if we need the value to be 1 or 0
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
        print(f"Total Voronoi centers: {centers.shape[0]}")

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
        
        # def get_region_centers_from_vertices(vor):
        #     centers = []
        #     for region in vor.regions:
        #         # Skip empty regions or regions with -1 (infinite vertices)
        #         if not region or -1 in region:
        #             continue

        #         # Get vertices of the region
        #         vertices = vor.vertices[region]
        #         # Calculate centroid
        #         center = vertices.mean(axis=0)
        #         centers.append(center)

        #     return np.array(centers)
            
        # centers = get_region_centers_from_vertices(vor)
        # centers = np.array(vor.vertices, dtype=int)

        # Step 5: Plot the occupancy grid
        plt.imshow(self.occupancy_grid, cmap='gray', origin='lower')
        plt.scatter(centers[:, 1], centers[:, 0], c='r', label='Voronoi centers')
        for i, center in enumerate(centers):
            plt.text(center[1] + 5, center[0] + 5, str(i), color='blue', fontsize=6)  # Annotate node IDs
        
        # Step 6: Plot the Voronoi diagram (optional)
        # voronoi_plot_2d(vor, ax=plt.gca(), show_vertices=False, line_colors='b', line_width=1)

        # Step 7: Construct and plot the MST (minimum spanning tree) for the Voronoi centers
        mst_edges = self.plot_mst(centers)
        plt.show()
        
        # Step 8: Save the Graph in the required format
        generate_graph(centers=centers, mst_edges=mst_edges, filename=f"{self.pgm_file.split('.')[0]}.graph", self.resolution, self.origin)
        
    def plot_mst(self, centers):
        # Compute pairwise distances between Voronoi centers
        dist_matrix = cdist(centers, centers, 'euclidean')

        # Use Kruskal's algorithm (via scipy minimum spanning tree function)
        mst = minimum_spanning_tree(dist_matrix)
        print(f"Minimum Spanning Tree (MST) shape: {mst.shape}")
        # Convert the sparse matrix to a dense matrix for easier processing
        mst = mst.toarray()
        
        # Plot the MST by connecting the Voronoi centers with pathfinding checks
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                if mst[i, j] > 0:  # If there is an edge
                    start = tuple(centers[i])
                    end = tuple(centers[j])

                    # Use A* to check if a path exists
                    path = self.astar.a_star(start, end)

                    if path:
                        # If a valid path exists, plot the path
                        path = np.array(path)
                        plt.plot(path[:, 1], path[:, 0], 'g-', lw=1)
                        
        # Step 4: Extract edges from the MST and calculate the distance
        mst_edges = []
        for i in range(mst.shape[0]):
            for j in range(i + 1, mst.shape[1]):
                if mst[i, j] > 0:  # If there's an edge between node i and node j
                    mst_edges.append((i, j, mst[i, j]))  # Add (node1, node2, distance)

        return mst_edges

if __name__ == "__main__":
    G = patrol_graph('map_name.yaml', 'map_name.pgm')
    # G = patrol_graph('map1.yaml', 'map1.pgm')
    h = G.height
    w = G.width
    # print(h, w)
    distance_threshold = (h*w)//10000
    print("Distance Threshold: ", distance_threshold)
    dilation_radius = 2 * distance_threshold
    G.gen_voronoi(distance_threshold=distance_threshold, dilation_radius=dilation_radius)