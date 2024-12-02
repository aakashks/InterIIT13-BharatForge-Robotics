import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import networkx as nx
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from skimage.feature import peak_local_max

def load_pgm_map(pgm_path):
    """Load PGM file and convert to binary occupancy grid"""
    img = Image.open(pgm_path)
    img_array = np.array(img)
    # Convert to binary (0 for free space, 1 for obstacles)
    occupancy_grid = (img_array < 128).astype(int)
    return occupancy_grid

def find_navigation_points(occupancy_grid):
    """Find navigation points using distance transform and local maxima"""
    # Calculate distance transform
    dist_transform = distance_transform_edt(1 - occupancy_grid)

    # Find local maxima
    min_distance = 10  # Minimum distance between points
    threshold_abs = 5  # Minimum distance from obstacles
    coordinates = peak_local_max(dist_transform, 
                               min_distance=min_distance,
                               threshold_abs=threshold_abs)

    return coordinates

def create_navigation_graph(coordinates, occupancy_grid):
    """Create navigation graph from points"""
    G = nx.Graph()

    # Add nodes
    for i, (y, x) in enumerate(coordinates):
        G.add_node(i, pos=(x, y))

    # Connect nodes based on visibility and distance
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            y1, x1 = coordinates[i]
            y2, x2 = coordinates[j]

            # Check if distance is reasonable
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if dist > 200:  # Skip if too far
                continue

            # Check line of sight
            x_line = np.linspace(x1, x2, int(dist))
            y_line = np.linspace(y1, y2, int(dist))
            x_line = x_line.astype(int)
            y_line = y_line.astype(int)

            # If path doesn't cross obstacles, add edge
            if not np.any(occupancy_grid[y_line, x_line]):
                G.add_edge(i, j)

    return G

def plot_graph_on_map(occupancy_grid, graph):
    """Plot the navigation graph on top of the occupancy grid"""
    plt.figure(figsize=(12, 12))

    # Plot occupancy grid
    plt.imshow(occupancy_grid, cmap='binary')

    # Get node positions
    pos = nx.get_node_attributes(graph, 'pos')

    # Plot edges
    for edge in graph.edges():
        x1, y1 = pos[edge[0]]
        x2, y2 = pos[edge[1]]
        plt.plot([x1, x2], [y1, y2], 'r-', linewidth=1)

    # Plot nodes
    for node, (x, y) in pos.items():
        plt.plot(x, y, 'bo', markersize=8)
        plt.text(x+5, y+5, str(node), color='blue', fontsize=8)

    plt.title('Navigation Graph on Occupancy Grid')
    plt.grid(True)
    plt.show()

def main(pgm_path):
    # Load map
    occupancy_grid = load_pgm_map(pgm_path)

    # Find navigation points
    coordinates = find_navigation_points(occupancy_grid)

    # Create navigation graph
    graph = create_navigation_graph(coordinates, occupancy_grid)

    # Plot result
    plot_graph_on_map(occupancy_grid, graph)

# Example usage
if __name__ == "__main__":
    pgm_file = "map_name.pgm"  # Replace with your PGM file path
    main(pgm_file)
