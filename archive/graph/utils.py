import numpy as np
import heapq

class AStar:
    def __init__(self, grid):
        self.grid = grid
        self.rows = grid.shape[0]
        self.cols = grid.shape[1]
    
    def is_valid(self, x, y):
        # Check if the position is within bounds and not an obstacle
        return 0 <= x < self.rows and 0 <= y < self.cols and self.grid[x, y] == 0

    def a_star(self, start, end):
        # A* algorithm to find the shortest path from start to end
        # Start and end are tuples (x, y)
        
        open_list = []
        heapq.heappush(open_list, (0, start))  # (f, position)
        
        g_costs = {start: 0}  # Cost from start to current node
        f_costs = {start: self.heuristic(start, end)}  # f = g + h
        
        came_from = {}  # For reconstructing the path
        
        while open_list:
            _, current = heapq.heappop(open_list)
            
            if current == end:
                # Reconstruct the path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]  # Reverse the path to get it from start to end
            
            # Explore neighbors
            neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 4 directions
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if self.is_valid(neighbor[0], neighbor[1]):
                    tentative_g = g_costs[current] + 1  # Uniform cost for each move
                    
                    if neighbor not in g_costs or tentative_g < g_costs[neighbor]:
                        came_from[neighbor] = current
                        g_costs[neighbor] = tentative_g
                        f_costs[neighbor] = tentative_g + self.heuristic(neighbor, end)
                        heapq.heappush(open_list, (f_costs[neighbor], neighbor))
        
        return []  # No path found
    
    def heuristic(self, current, end):
        # Use Manhattan distance as the heuristic (you can also use Euclidean)
        return abs(current[0] - end[0]) + abs(current[1] - end[1])

# Function to calculate relative direction (N, S, E, W) between two points
def get_direction(curr, neighbor):
    x_diff = neighbor[0] - curr[0]
    y_diff = neighbor[1] - curr[1]

    if abs(x_diff) > abs(y_diff):  # More horizontal movement
        if x_diff > 0:
            return 'E'  # East
        else:
            return 'W'  # West
    else:  # More vertical movement
        if y_diff > 0:
            return 'N'  # North
        else:
            return 'S'  # South
        
def generate_graph(centers, mst_edges, filename):
    # centers: List of Voronoi centers as (x, y) coordinates
    # mst_edges: List of edges in MST [(center1, center2, distance), ...]

    # Create a dictionary to hold node information
    graph = {}

    # Add each center as a node with its neighbors
    for i, center in enumerate(centers):
        graph[i] = {
            'coordinates': center,
            'neighbors': []
        }
    
    # Now add edges between nodes (neighbors)
    for edge in mst_edges:
        node1, node2, distance = edge
        
        # Get the relative direction from node1 to node2
        direction = get_direction(centers[node1], centers[node2])
        
        # Add the neighbor information for both nodes
        graph[node1]['neighbors'].append({
            'neighbor_id': node2,
            'direction': direction,
            'distance': distance
        })
        
        # Do the same for the reverse direction
        reverse_direction = get_direction(centers[node2], centers[node1])
        graph[node2]['neighbors'].append({
            'neighbor_id': node1,
            'direction': reverse_direction,
            'distance': distance
        })
    
    # Open the file in write mode
    with open(filename, 'w') as f:
        # Output the graph in the required format
        for node_id, node_data in graph.items():
            # Node ID, X, Y coordinates
            f.write(f"{node_id}\n")
            f.write(f"{node_data['coordinates'][0]}\n")  # X coordinate
            f.write(f"{node_data['coordinates'][1]}\n")  # Y coordinate
            
            # Number of neighbors (k)
            k = len(node_data['neighbors'])
            f.write(f"{k}\n")
            
            # List neighbors with their directions and distances
            for neighbor in node_data['neighbors']:
                f.write(f"{neighbor['neighbor_id']}\n")
                f.write(f"{neighbor['direction']}\n")
                f.write(f"{neighbor['distance']}\n")
            f.write('\n')