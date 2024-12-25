# ROS2 and related imports
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image as msgimg
from std_msgs.msg import String, Float64MultiArray, MultiArrayDimension
from visualization_msgs.msg import MarkerArray, Marker
# from patrolling_interfaces.srv import SendRobotToObject  # Custom service import

# Utility imports
import random
import math
import os
import json
import threading
from datetime import datetime
from functools import partial
from typing import List, Dict, Any
import logging
import requests
import yaml
import numpy as np
from PIL import Image
import cv2
from tf_transformations import quaternion_from_euler
from cv_bridge import CvBridge
from icecream import ic
from dotenv import load_dotenv
from collections import deque

# Custom module imports
from .robot_manager import RobotManager
from .points_to_monitor import get_random_points_to_monitor_near_obstacles2, Atransform, InverseAtransform

load_dotenv()

user = os.getenv('USER')

if not os.getenv('DATA_DIR'):
    os.makedirs(f'/home/{user}/data', exist_ok=True)
    os.environ['DATA_DIR'] = f'/home/{user}/data'


########

import requests
from typing import List, Dict, Any
import base64
import logging

class EmbeddingClient:
    def __init__(self, base_url: str = "http://0.0.0.0:8000"):
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)

    def update_db(self, 
                 pose_key: str,
                 image_path: str,
                 robot_name: str,
                 timestamp: str,
                 depth_image_path: str,
                 pose: Dict[str, float | int]):
        """
        Update the database with new pose data
        """

        with open(image_path, 'rb') as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        
        data = {
            "pose_key": pose_key,
            "image_path": image_path,
            "image_b64": base64_image,
            "robot_name": robot_name,
            "timestamp": timestamp,
            "depth_image_path": depth_image_path,
            "pose": pose
        }

        try:
            response = requests.post(f"{self.base_url}/update_db", json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error updating: {str(e)}")
            raise

    def query_db(self, prompts: List[str], limit: int = 10):
        """
        Query the database with prompts
        """
        try:
            response = requests.post(f"{self.base_url}/query_db", 
                                  json={"prompts": prompts, "limit": limit})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error querying: {str(e)}")
            raise


# Assuming RobotManager and other necessary classes/functions are imported properly
# from your_custom_module import RobotManager, Atransform, get_random_points_to_monitor_near_obstacles2, ImageLoader, OpenCLIPEmbeddingFunction

class GoalAssignerNode(Node):

    def __init__(self):
        super().__init__('goal_assigner')

        self.db_client = EmbeddingClient(os.getenv('DB_URL', 'http://192.168.124.197:8000'))

        # Default parameters
        self.declare_parameter('robot_names', ['robot1'])
        self.declare_parameter('world_frame', 'map')
        self.declare_parameter('status_check_interval', 2.0)  # in seconds
        self.declare_parameter('map_yaml_file', 'config/map.yaml')
        self.declare_parameter('map_file', 'config/map.pgm')
        self.declare_parameter('objects_file', 'config/objects.yaml')  # Parameter for objects file

        # Get parameters
        self.robot_names = self.get_parameter('robot_names').get_parameter_value().string_array_value
        self.world_frame = self.get_parameter('world_frame').get_parameter_value().string_value
        self.status_check_interval = self.get_parameter('status_check_interval').get_parameter_value().double_value
        self.map_yaml_file = self.get_parameter('map_yaml_file').get_parameter_value().string_value
        self.map_file = self.get_parameter('map_file').get_parameter_value().string_value
        self.objects_file = self.get_parameter('objects_file').get_parameter_value().string_value  # Get objects file


        callback_group1 = MutuallyExclusiveCallbackGroup()  
        callback_group2 = MutuallyExclusiveCallbackGroup()  
        callback_group3 = MutuallyExclusiveCallbackGroup()
        callback_group4 = MutuallyExclusiveCallbackGroup()
        callback_group5 = MutuallyExclusiveCallbackGroup()
        # Load the map metadata
        with open(self.map_yaml_file, 'r') as yaml_file:
            map_yaml = yaml.safe_load(yaml_file)
        self.resolution = map_yaml['resolution']
        self.origin = map_yaml['origin']

        # Load the occupancy grid
        self.occupancy_grid = self.pgm2occupancy(self.map_file)
        self.width, self.height = self.occupancy_grid.shape
        self.get_logger().info(f"{self.origin, self.height, self.width}")

        # Load monitoring points
        # TODO: TO RUN AFTER A CERTAIN TIME PERIODICALLY
        sampled_grid_points = get_random_points_to_monitor_near_obstacles2(occupancy_grid=self.occupancy_grid)
        self.monitoring_points = sampled_grid_points
        sampled_grid_points = Atransform(sampled_grid_points, self.occupancy_grid.shape, self.get_logger())
        self.monitoring_points_world = self.convert_grid_to_world(sampled_grid_points)
        self.get_logger().info(f"Loaded {len(self.monitoring_points)} monitoring points.")
        # Initialize RobotManager with occupancy grid and monitoring points
        self.robot_manager = RobotManager(self.occupancy_grid, self.monitoring_points, logger=self.get_logger())

        # Initialize shared components
        self.bridge = CvBridge()

        # Initialize robot data structures
        self.robots = {}

        # Timer for periodic status checks
        self.status_check_timer = self.create_timer(
            0.2, self.check_goal_statuses ,callback_group=callback_group1
        )
        self.sending_robot = self.create_subscription(String,"coordinate_data",self.listener_callback,10,callback_group=callback_group4)
        self.get_logger().info("Subscribed to coordinate_data")

        # self.print_positions_timer = self.create_timer(
        #     1.0, self.update_latest_postion)
        # Create the service to send robots to objects
        # self.send_robot_service = self.create_service(
        #     SendRobotToObject,
        #     'send_robot_to_object',
        #     self.send_robot_to_object_callback
        # )
        self.latest_positions = {robot: None for robot in self.robot_names}

        self.odom_subscribers = {}

        for robot_name in self.robot_names:
            action_topic = f'/{robot_name}/navigate_to_pose'
            action_client = ActionClient(self, NavigateToPose, action_topic)

            # Assign a unique color to each robot
            color = self.assign_unique_color(robot_name)

            # Create a Marker publisher for each robot with a robot-specific topic
            marker_topic = f'/{robot_name}/monitoring_points_marker'
            marker_topic_point = f'/{robot_name}/monitoring_points_marker_point'

            marker_publisher = self.create_publisher(MarkerArray, marker_topic, 10)
            marker_publisher_point = self.create_publisher(MarkerArray, marker_topic_point, 10)

            # Add per-robot variables to the robots dictionary
            self.robots[robot_name] = {
                'action_client': action_client,
                'current_point': None,
                'assigned_point': None,
                'goal_handle': None,
                'status': GoalStatus.STATUS_UNKNOWN,  # Initialize with UNKNOWN status
                'marker_publisher': marker_publisher,  # Publisher for robot's monitoring points
                'marker_publisher_point': marker_publisher_point,
                'color': color,  # Unique color for the robot
                'monitoring_points': [],  # List to store this robot's monitoring points
                # Per-robot image data and locks
                'image_lock': threading.Lock(),
                'latest_image': None,
                'latest_depth_image': None
            }

            # Create per-robot subscribers for image data
            image_data_sub = self.create_subscription(
                msgimg,  # Replace with your actual Image message type
                f'/{robot_name}/image_data',
                partial(self.image_callback, robot_name=robot_name),
                2  ,
                callback_group=callback_group5
            )
            depth_image_data_sub = self.create_subscription(
                Float64MultiArray,  # Replace with your actual depth image message type
                f'/{robot_name}/depth_image_data',
                partial(self.depth_image_callback, robot_name=robot_name),
                2,
                callback_group=callback_group5
            )

            # Store the subscribers in the robots dictionary (optional)
            self.robots[robot_name]['image_data_sub'] = image_data_sub
            self.robots[robot_name]['depth_image_data_sub'] = depth_image_data_sub

            topic_name = f'/{robot_name}/odom'
            self.odom_subscribers[robot_name] = self.create_subscription(
                Odometry,
                topic_name,
                partial(self.odom_callback, robot_name=robot_name),
                10,
                callback_group=callback_group5
            )
            self.get_logger().info(f'Subscribed to {topic_name}')
        
        # Initialize robots in RobotManager
        for robot_name in self.robot_names:
            initial_position = self.get_random_free_position()
            grid_position = self.world_to_grid(initial_position)
            self.robot_manager.add_robot(robot_name, 'patrolling', initial_position=grid_position)
            self.get_logger().info(f'Added robot {robot_name} at grid position {grid_position}')

        # Divide the available space among robots
        self.robot_manager.launch()

        # Assign monitoring points to robots
        self.robot_manager.assign_monitoring_points_to_robots()

        # Retrieve and store each robot's monitoring points
        for robot_name in self.robot_names:
            robot_obj = self.robot_manager.get_robot_by_id(robot_name)
            if robot_obj:
                world_points = robot_obj.assigned_points  # Assuming this attribute exists
                self.robots[robot_name]['monitoring_points'] = world_points
                self.get_logger().info(f'Assigned {len(world_points)} monitoring points to {robot_name}')

        # Publisher for current points
        self.current_points_publisher = self.create_publisher(String, 'current_points', 10)
        self.publish_current_points_timer = self.create_timer(2.0, self.publish_current_points, callback_group=callback_group2)

        # Load objects from the configuration file
        self.load_objects(self.objects_file)

        # ------------------- Integration for Rviz Visualization -------------------

        # Timer to publish markers periodically
        self.marker_publish_rate_each = 3.0  # seconds
        self.marker_publish_timer_each = self.create_timer(self.marker_publish_rate_each, self.publish_markers_each, callback_group=callback_group3)

        # Publisher for monitoring points markers
        self.pub_marker_points = self.create_publisher(MarkerArray, 'visualization_marker_array_points', 10)

        # Timer to publish markers periodically
        self.marker_publish_rate = 3.0  # seconds
        self.marker_publish_timer = self.create_timer(self.marker_publish_rate, self.publish_markers, callback_group=callback_group3)
        # Flag to ensure markers are published only once if the points don't change
        self.markers_published = False

        # --------------------------------------------------------------------------

        # ------------------- Integration for Robot-Specific Markers -------------------

        # Timer to publish robot markers periodically
        self.robot_marker_publish_rate = 3.0  # seconds
        self.robot_marker_publish_timer = self.create_timer(self.robot_marker_publish_rate, self.publish_robot_markers,callback_group=callback_group3)

        # --------------------------------------------------------------------------
        # Create subscribers for each robot's odom

        # Optionally, set up a timer to periodically print the latest positions

    def odom_callback(self, msg: Odometry, robot_name: str):
        """
        Callback function for odometry messages.
        Stores the latest position of the robot.
        """
        position = msg.pose.pose.position
        self.latest_positions[robot_name] = position
        self.get_logger().debug(
            f'Received odom for {robot_name}: Position -> x: {position.x}, y: {position.y}, z: {position.z}'
        )

    def update_latest_postion(self):
        self.get_logger().info(f"No position received yet for ")
        for robot_name, position in self.latest_positions.items():
            if position:
                self.get_logger().info(f"Latest position for {robot_name}: {position.x}, {position.y}")
            # else:
        
    def image_callback(self, msg, robot_name):
        """Callback function for image data"""
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Access the robot's data
            robot = self.robots[robot_name]

            # Store the image thread-safely
            with robot['image_lock']:
                robot['latest_image'] = cv_image.copy()

            # Optional: Display the image
            # cv2.imshow(f"{robot_name} Camera Image", cv_image)
            # cv2.waitKey(1)

            # self.get_logger().debug(f"Received image from {robot_name} with shape: {cv_image.shape}")

        except Exception as e:
            self.get_logger().error(f'Error processing image from {robot_name}: {str(e)}')

    def depth_image_callback(self, msg, robot_name):
        """Callback function for depth image data"""
        try:
            # Get the dimensions from the message layout
            height = msg.layout.dim[0].size
            width = msg.layout.dim[1].size
            # Convert the flat array back to 2D
            depth_data = np.array(msg.data)

            # Replace the special value with NaN
            depth_data[depth_data == -999.999] = np.nan

            # Reshape to original dimensions
            depth_image = depth_data.reshape((height, width))

            # Access the robot's data
            robot = self.robots[robot_name]

            # Store the depth image thread-safely
            with robot['image_lock']:
                robot['latest_depth_image'] = depth_image.copy()

            # self.get_logger().debug(f"Received depth image from {robot_name} with shape: {depth_image.shape}")

        except Exception as e:
            self.get_logger().error(f'Error processing depth image from {robot_name}: {str(e)}')

    def assign_unique_color(self, robot_name):
        """
        Assigns a unique color to each robot based on its name.
        Customize the colors as needed.
        """
        color_map = {
            'robot1': (1.0, 0.0, 0.0, 1.0),  # Red
            'robot2': (0.0, 1.0, 0.0, 1.0),  # Green
            'robot3': (0.0, 0.0, 1.0, 1.0),  # Blue
            'robot4': (1.0, 1.0, 0.0, 1.0),  # Yellow
            'robot5': (1.0, 0.0, 1.0, 1.0),  # Magenta
            # Add more robots and colors as needed
        }
        return color_map.get(robot_name, (0.5, 0.5, 0.5, 1.0))  # Default to gray

    def convert_grid_to_world(self, grid_points):
        """
        Converts grid indices to world coordinates.
        :param grid_points: numpy array of shape (N, 2) with (row, col) indices
        :return: List of tuples [(x1, y1), (x2, y2), ...]
        """
        world_points = []
        for point in grid_points:
            row, col = point
            x = row * self.resolution + self.origin[0]
            y = self.origin[1] + col * self.resolution  # Corrected y-coordinate calculation
            world_points.append((x, y))
        return world_points

    def world_to_grid(self, world_point):
        """
        Converts world coordinates to grid indices.
        :param world_point: [x, y] in world coordinates
        :return: [row, col] in grid indices
        """
        x, y = world_point
        col = int((x - self.origin[0]) / self.resolution)
        row = int((y - self.origin[1]) / self.resolution)
        return [row, col]

    def load_monitoring_points(self):
        """
        Loads monitoring points. Currently, points are loaded during initialization.
        This method can be expanded to dynamically load points if needed.
        """
        pass  # Points are loaded during initialization

    def load_objects(self, objects_file):
        """
        Loads objects from a YAML file.
        :param objects_file: Path to the YAML file
        """
        try:
            with open(objects_file, 'r') as file:
                data = yaml.safe_load(file)
                self.objects = data.get('objects', {})
                # self.get_logger().info(f"Loaded objects from {objects_file}")
        except Exception as e:
            self.get_logger().error(f"Failed to load objects: {e}")
            self.objects = {}

    # Define BFS for multi-source pathfinding
    def bfs_multi_source(self, grid, start_positions, goal):
        """ Multi-source BFS to find the shortest path from any robot to the target point """
        
        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Queue for BFS (stores (x, y, cost))
        queue = deque([(start[0], start[1], 0) for start in start_positions])
        # start_positions=[list(arr) for arr in start_positions]
        # Set to track visited cells
        visited = set(start_positions)
        
        # While the queue is not empty, expand nodes
        while queue:
            x, y, cost = queue.popleft()
            
            # If we reached the goal
            if (x, y) == goal:
                return cost  # Return the length of the path (cost)
            
            # Explore neighbors
            for direction in directions:
                nx, ny = x + direction[0], y + direction[1]
                
                # Check if within bounds and not an obstacle
                if (0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and
                    (nx, ny) not in visited and grid[nx][ny] == 0):  # free space
                    visited.add((nx, ny))
                    queue.append((nx, ny, cost + 1))
        
        return None  # No path found

    # Multi-source BFS to find the closest robot
    def closest_robot_with_obstacles(self, robots, point, grid):
        """ Use multi-source BFS to find the closest robot to the target point """
        # Filter robots that are valid positions (free spaces)
        #ic(robots)
        robots_with_names = robots.copy()
        robots = [robo[1] for robo in robots]
        valid_robots = [(robot[0],robot[1]) for robot in robots if grid[robot[0], robot[1]] == 0]
        
        # Call BFS to find the shortest path length from multiple robots to the point
        min_path_length = float('inf')
        closest_robot = None
        
        # Multi-source BFS for all valid robots
        path_length = self.bfs_multi_source(grid, valid_robots, tuple(point))
        
        if path_length is not None and path_length < min_path_length:
            min_path_length = path_length
            closest_robot = valid_robots[0]  # In this case, take the first robot
        
        for name, robot in robots_with_names:
            if robot[0] == closest_robot[0] and robot[1] == closest_robot[1]:
                return name
        

    def listener_callback(self, msg):  
        # ic(msg.data)  
        coord_data = json.loads(msg.data)    
    
        # Extract coordinates from the first object (using string key '0')  
        first_object = coord_data['0']  # Get the first object using string key  
        first_point = first_object['points'][0]  # Get the first point    
    
        # Extract x and y coordinates    
        x = first_point['pose_x']  
        y = first_point['pose_y']  
    
        point = (x, y)  # Create tuple of coordinates    
        new_point = {'x': x, 'y': y}  # Create a new point dictionary  
        self.get_logger().info(f'Extracted coordinates: x={x}, y={y}')    
    
    # # Rest of the function remains the same...    
    #     with open("/home/user1/ros2_ws/src/patroling/config/objects.yaml", 'r') as file:
    #         data = yaml.safe_load(file)

    #     # Find the next object key
    #     existing_objects = data.get("objects", {})
    #     new_object_key = f"object{len(existing_objects) + 1}"

    #     # Add the new point under the new key
    #     data['objects'][new_object_key] = new_point

        # # Write back to the YAML file
        # with open("/home/user1/ros2_ws/src/patroling/config/objects.yaml", 'w') as file:
        #     yaml.safe_dump(data, file)
        robots = []
        for robot, position in self.latest_positions.items():
            if position:
                #ic(position, "position")
                robo_point = InverseAtransform(
                    np.array([position.x, position.y]).reshape(1, -1),
                    self.occupancy_grid.shape,
                    self.origin[0],
                    self.origin[1],
                    self.resolution
                )
                robots.append((robot, robo_point[0]))  # Store robot name with position

        #ic(point)
        point = np.array(point).reshape(1, -1)
        point = InverseAtransform(
            np.array(point),
            self.occupancy_grid.shape,
            self.origin[0],
            self.origin[1],
            self.resolution
        )
        #ic(point)

        nearest_robot = self.closest_robot_with_obstacles(
            robots, point[0], self.occupancy_grid
        )
        self.send_robot_to_object_callback(nearest_robot, new_point)

        
    def send_robot_to_object_callback(self, robot_name, position):


        # Create a PoseStamped goal
        pose = PoseStamped()
        pose.header.frame_id = self.world_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(position['x'])
        pose.pose.position.y = float(position['y'])
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0

        # Calculate yaw towards the nearest obstacle from the assigned point
        assigned_position = Point(
            x=pose.pose.position.x,
            y=pose.pose.position.y,
            z=0.0
        )
        nearest_object_position = self.get_nearest_object(assigned_position)
        if nearest_object_position:
            dx = nearest_object_position.x - assigned_position.x
            dy = nearest_object_position.y - assigned_position.y
            yaw = math.atan2(dy, dx)
            q = quaternion_from_euler(0.0, 0.0, yaw)
            pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            # self.get_logger().info(f"Calculated yaw for {robot_name}: {yaw} radians")
        else:
            pose.pose.orientation.w = 1.0  # Default orientation

        # Send goal to the robot
        self.send_goal_to_robot(robot_name, pose)

        # Update robot's state to 'working' to prevent patrol goal assignments
        robot_obj = self.robot_manager.get_robot_by_id(robot_name)
        if robot_obj:
            x, y = self.latest_positions[robot_name].x, self.latest_positions[robot_name].y
            robot_obj.position = x, y
            robot_obj.change_state('working')
            self.robot_manager._divide_available_space()

            self.get_logger().info(f"Robot {robot_name} state changed to 'working'.")


    def send_goal_to_robot(self, robot_name, pose):
        """
        Sends a navigation goal to a specified robot.
        """
        robot = self.robots[robot_name]
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        action_client = robot['action_client']
        action_topic = f'/{robot_name}/navigate_to_pose'

        if not action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f'Action server not available for {robot_name} at {action_topic}')
            return

        self.get_logger().info(f'Sending external goal to {robot_name}')
        send_goal_future = action_client.send_goal_async(
            goal_msg,
            feedback_callback=lambda feedback: self.feedback_callback(robot_name, feedback)
        )
        send_goal_future.add_done_callback(lambda future: self.goal_response_callback(robot_name, future))

        # Update robot status to reflect that it's executing an external goal
        robot['status'] = GoalStatus.STATUS_ACCEPTED
        robot['assigned_point'] = pose.pose.position  # Assign the external goal
        robot['current_point'] = None

    def pgm2occupancy(self, pgm_file):
        """
        Converts a PGM file to an occupancy grid.
        Free space: 0
        Obstacles: 1
        """
        img = Image.open(pgm_file)
        img = img.convert('L')  # 'L' mode is for grayscale
        img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
        occupancy_grid = np.zeros_like(img_array, dtype=int)
        occupancy_grid[img_array > 0.9] = 0  # Free space
        occupancy_grid[img_array <= 0.9] = 1  # Obstacles
        return occupancy_grid

    def get_random_free_position(self):
        """
        Retrieves a random free position from the occupancy grid.
        :return: [x, y] in world coordinates
        """
        free_space_indices = np.argwhere(self.occupancy_grid == 0)
        if len(free_space_indices) == 0:
            self.get_logger().error('No free space available in the occupancy grid.')
            return [0, 0]
        random_index = random.choice(free_space_indices)
        row, col = random_index
        x = col * self.resolution + self.origin[0]
        y = self.origin[1] + row * self.resolution  # Corrected y-coordinate calculation
        return [x, y]

    def publish_current_points(self):
        """
        Publishes the current points and their assignments to a ROS topic.
        """
        status_messages = []
        for robot_name, data in self.robots.items():
            status = data['status']
            status_str = self.get_status_string(status) if status is not None else 'NO_STATUS'
            current_point = data['current_point']
            assigned_point = data['assigned_point']

            status_messages.append(
                f'{robot_name}: current={current_point}, '
                f'assigned={assigned_point}, status={status_str}'
            )

        msg = String()
        msg.data = ' | '.join(status_messages)
        self.current_points_publisher.publish(msg)

    def check_goal_statuses(self):
        """
        Timer callback to periodically check the status of each robot's goal.
        Assign new goals if necessary based on the current status.
        """
        for robot_name, data in self.robots.items():
            status = data['status']
            goal_handle = data['goal_handle']

            # Get robot object
            robot_obj = self.robot_manager.get_robot_by_id(robot_name)
            if not robot_obj:
                self.get_logger().error(f'Robot {robot_name} not found in RobotManager.')
                continue

            # Skip robots that are not in 'patrolling' state
            if robot_obj.state != 'patrolling':
                self.get_logger().debug(f'Robot {robot_name} is in state {robot_obj.state}. Skipping goal assignment.')
                continue

            # Log current status
            status_str = self.get_status_string(status) if status is not None else 'NO_STATUS'
            self.get_logger().debug(f'[{robot_name}] Current goal status: {status_str}')

            if status in [
                GoalStatus.STATUS_SUCCEEDED,
                GoalStatus.STATUS_ABORTED,
                GoalStatus.STATUS_CANCELED,
                GoalStatus.STATUS_UNKNOWN,
            ]:
                self.get_logger().info(f'Assigning new patrol goal to {robot_name}')
                self.assign_new_monitoring_goal(robot_name)

            elif status in [GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING]:
                # Robot is on the way; do nothing
                pass
            else:
                # Handle other statuses if necessary
                pass

    def saving_data(self, robot_name, robot_obj):
        x, y, z, xo, yo, zo, wo = robot_obj.current_goal        
        os.makedirs(os.getenv('DATA_DIR'), exist_ok=True)
        os.makedirs(f"{os.getenv('DATA_DIR')}/image", exist_ok=True)
        os.makedirs(f"{os.getenv('DATA_DIR')}/depth", exist_ok=True)

        # Access the robot's data
        robot = self.robots[robot_name]
        with robot['image_lock']:
            latest_image = robot['latest_image']
            latest_depth_image = robot['latest_depth_image']

        if (latest_depth_image is not None) and (latest_image is not None):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            image_path = f"{os.getenv('DATA_DIR')}/image/images_{robot_name}_{timestamp}.jpg"
            depth_npz_path = f"{os.getenv('DATA_DIR')}/depth/depth_data_{robot_name}_{timestamp}.npz"  # Updated to NPZ

            # Save RGB image as JPG
            cv2.imwrite(image_path, latest_image)


            # Save depth data as NPZ
            np.savez(depth_npz_path, depth=latest_depth_image)


            json_file_path = f"{os.getenv('DATA_DIR')}/data.json"
            json_data = {}

            # Read existing JSON data if file exists
            if os.path.exists(json_file_path):
                with open(json_file_path, "r") as f:
                    try:
                        json_data = json.load(f)
                    except json.JSONDecodeError:
                        json_data = {}

            # Create pose key
            pose_key = f"{x:.3f}_{y:.3f}_{z:.3f}_{xo:.3f}_{yo:.3f}_{zo:.3f}_{wo:.3f}"

            # Create or update the pose entry
            if pose_key not in json_data:
                json_data[pose_key] = {
                    "pose": {
                        "x": x,
                        "y": y,
                        "z": z,
                        "xo": xo,
                        "yo": yo,
                        "zo": zo,
                        "w": wo
                    },
                    "data": []
                }

            # Add new image data
            image_data = {
                "timestamp": timestamp,
                "robot_name": robot_name,
                "image_path": image_path,
                "depth_data_path": depth_npz_path  # Reference to NPZ file
            }

            json_data[pose_key]["data"].append(image_data)

            pose_data = {
                'x': x,
                'y': y,
                'z': z,
                'xo': xo,
                'yo': yo,
                'zo': zo,
                'w': wo
            }

            
            self.db_client.update_db(
                pose_key=pose_key,
                timestamp=timestamp,
                robot_name=robot_name,
                image_path=image_path,
                depth_image_path=depth_npz_path,
                pose=pose_data
            )


            # Write updated data back to JSON file
            with open(json_file_path, "w") as f:
                json.dump(json_data, f, indent=4)


        else:
            if latest_depth_image is None:
                self.get_logger().error(f'Skipping {robot_name}: latest_depth_image is None')
            if latest_image is None:
                self.get_logger().error(f'Skipping {robot_name}: latest_image is None')

    def assign_new_monitoring_goal(self, robot_name):
        """
        Assigns a new monitoring point to the robot.
        """
        robot_obj = self.robot_manager.get_robot_by_id(robot_name)
        if not robot_obj:
            self.get_logger().error(f'Robot {robot_name} not found in RobotManager.')
            return

        # Assuming robot_obj.get_next_monitoring_point() returns a list or tuple
        next_point = np.asarray(robot_obj.get_next_monitoring_point())
        next_point = next_point.reshape(1, -1)

        # Now perform the transformation
        next_point = self.convert_grid_to_world(
            Atransform(next_point, self.occupancy_grid.shape, self.get_logger())
        )

        if not next_point:
            self.get_logger().warn(f'No more monitoring points to assign for {robot_name}.')
            return

        pose = PoseStamped()
        pose.header.frame_id = self.world_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(next_point[0][0])
        pose.pose.position.y = float(next_point[0][1])
        pose.pose.position.z = 0.0

        # Calculate yaw towards the nearest obstacle from the assigned point
        assigned_position = Point(
            x=pose.pose.position.x,
            y=pose.pose.position.y,
            z=0.0
        )
        nearest_object_position = self.get_nearest_object(assigned_position)
        if nearest_object_position:
            dx = nearest_object_position.x - assigned_position.x
            dy = nearest_object_position.y - assigned_position.y
            yaw = math.atan2(dy, dx)
            q = quaternion_from_euler(0.0, 0.0, yaw)
            pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            # self.get_logger().info(f"Calculated yaw for {robot_name}: {yaw} radians")
        else:
            pose.pose.orientation.w = 1.0  # Default orientation

        # Log the coordinates of the goal position
        self.get_logger().info(f'Goal position for {robot_name}: x={pose.pose.position.x}, y={pose.pose.position.y}, z={pose.pose.position.z}')

        # Create a goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        action_client = self.robots[robot_name]['action_client']
        action_topic = f'/{robot_name}/navigate_to_pose'

        # Wait until the action server is available
        if not action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f'Action server not available for {robot_name} at {action_topic}')
            return

        # Send goal
        self.get_logger().info(f'Sending patrol goal to {robot_name}: Point {next_point}')
        send_goal_future = action_client.send_goal_async(
            goal_msg,
            feedback_callback=lambda feedback: self.feedback_callback(robot_name, feedback)
        )
        send_goal_future.add_done_callback(lambda future: self.goal_response_callback(robot_name, future))
        robot_obj.current_goal = [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z, pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w]
        # Update robot status
        self.robots[robot_name]['status'] = GoalStatus.STATUS_EXECUTING
        self.robots[robot_name]['assigned_point'] = pose.pose.position  # Assign the patrol goal
        self.robots[robot_name]['current_point'] = None

    def goal_response_callback(self, robot_name, future):
        """
        Callback after a goal is sent to a robot.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info(f'Goal rejected for {robot_name}')
            self.robots[robot_name]['status'] = GoalStatus.STATUS_UNKNOWN
            self.robots[robot_name]['goal_handle'] = None
            return

        self.get_logger().info(f'Goal accepted for {robot_name}')
        self.robots[robot_name]['goal_handle'] = goal_handle
        self.robots[robot_name]['status'] = GoalStatus.STATUS_ACCEPTED

        # Get result asynchronously
        goal_handle.get_result_async().add_done_callback(lambda future: self.get_result_callback(robot_name, future))

    def get_result_callback(self, robot_name, future):
        """
        Callback to handle the result of a goal.
        """
        result = future.result().result
        status = future.result().status

        # Update robot's status
        self.robots[robot_name]['status'] = status

        status_str = self.get_status_string(status)
        self.get_logger().info(f'Goal status for {robot_name}: {status_str}')

        if status == GoalStatus.STATUS_SUCCEEDED:
            # Save data for the robot
            robot_obj = self.robot_manager.get_robot_by_id(robot_name)
            if robot_obj:
                self.get_logger().info(f'Saving data for {robot_name}')
                self.saving_data(robot_name=robot_name, robot_obj=robot_obj)

            point = self.robots[robot_name]['assigned_point']
            self.get_logger().info(f'Goal succeeded for {robot_name}: Point {point}')
            self.robots[robot_name]['current_point'] = point

            # Reset assigned point
            self.robots[robot_name]['assigned_point'] = None

            # Change robot state back to 'patrolling'
            if robot_obj:
                if not self.latest_positions[robot_name]:
                    self.get_logger().error(f'No latest position available for {robot_name}')
                    return
                x, y = self.latest_positions[robot_name].x, self.latest_positions[robot_name].y
                robot_obj.position = x, y
                if robot_obj.state == "working":
                    robot_obj.change_state('patrolling')
                    self.robot_manager._divide_available_space()

                self.get_logger().info(f"Robot {robot_name} state changed back to 'patrolling'.")

            # **New Addition:** Immediately assign a new goal after completing the current one
            self.assign_new_monitoring_goal(robot_name)

        elif status in [GoalStatus.STATUS_CANCELED]:
            self.get_logger().warn(f'Goal for {robot_name} failed with status: {status_str}')

            # Change robot state back to 'patrolling' even if the goal failed
            robot_obj = self.robot_manager.get_robot_by_id(robot_name)
            if robot_obj:
                x, y = self.latest_positions[robot_name].x, self.latest_positions[robot_name].y
                robot_obj.position = x, y
                # robot_obj.change_state('patrolling')

                # self.get_logger().info(f"Robot {robot_name} state changed back to 'patrolling'.")

            # **New Addition:** Immediately assign a new goal even if the previous one failed
            self.assign_new_monitoring_goal(robot_name)

        else:
            self.get_logger().info(f'Goal for {robot_name} ended with status: {status_str}')

        # Reset the goal_handle to indicate that the robot is ready for a new goal
        self.robots[robot_name]['goal_handle'] = None

    def feedback_callback(self, robot_name, feedback_msg):
        """
        Optional: Process feedback from the action server.
        """
        # Example: Log the progress
        # self.get_logger().info(f'Feedback from {robot_name}: {feedback_msg.feedback}')
        pass

    def get_status_string(self, status):
        """
        Converts status code to human-readable string.
        """
        status_dict = {
            GoalStatus.STATUS_UNKNOWN: 'UNKNOWN',
            GoalStatus.STATUS_ACCEPTED: 'ACCEPTED',
            GoalStatus.STATUS_EXECUTING: 'EXECUTING',
            GoalStatus.STATUS_CANCELING: 'CANCELING',
            GoalStatus.STATUS_SUCCEEDED: 'SUCCEEDED',
            GoalStatus.STATUS_CANCELED: 'CANCELED',
            GoalStatus.STATUS_ABORTED: 'ABORTED'
        }
        return status_dict.get(status, f'UNKNOWN_STATUS_{status}')

    # ------------------- Marker Publishing for Rviz -------------------

    def publish_markers_each(self):
        """
        Publishes the monitoring points for each robot as separate MarkerArray messages with unique colors.
        """
        for robot_name, data in self.robots.items():
            robot_obj = self.robot_manager.get_robot_by_id(robot_name)

            marker_array = MarkerArray()
            current_time = self.get_clock().now().to_msg()

            marker = Marker()
            marker.header.frame_id = self.world_frame
            marker.header.stamp = current_time
            marker.ns = f"{robot_name}_monitoring_points"
            marker.id = 0
            marker.type = Marker.POINTS
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2  # Size of the points
            marker.scale.y = 0.2
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = data['color']

            # Add each monitoring point to the marker
            # ic(robot_obj.assigned_points)
            for point in self.convert_grid_to_world(Atransform(np.array(robot_obj.assigned_points), self.occupancy_grid.shape, self.get_logger())):
                point_msg = Point()
                point_msg.x = float(point[0])
                point_msg.y = float(point[1])
                point_msg.z = 0.0
                marker.points.append(point_msg)

            marker_array.markers.append(marker)

            # Publish the marker array to the robot-specific topic
            data['marker_publisher'].publish(marker_array)

    def publish_markers(self):
        """
        Publishes the monitoring points as MarkerArray messages for Rviz visualization.
        This function is called periodically by a timer.
        """
        marker_array = MarkerArray()
        current_time = self.get_clock().now().to_msg()

        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = current_time
        marker.ns = "monitoring_points"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1  # Size of the points
        marker.scale.y = 0.1
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # Fully opaque
        marker.points = []

        for point in self.monitoring_points_world:
            point_msg = Point()
            point_msg.x = float(point[0])
            point_msg.y = float(point[1])
            point_msg.z = 0.0
            marker.points.append(point_msg)

        marker_array.markers.append(marker)

        # Publish the markers
        self.pub_marker_points.publish(marker_array)

    def publish_robot_markers(self):
        """
        Publishes markers for each robot on their respective topics with unique colors.
        The marker will point towards the nearest object.
        """
        for robot_name, data in self.robots.items():
            # Create a separate MarkerArray for each robot
            marker_array = MarkerArray()

            marker = Marker()
            marker.header.frame_id = self.world_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = f"{robot_name}_namespace"
            marker.id = 0  # Unique ID per robot
            marker.type = Marker.ARROW  # Use ARROW to indicate direction
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 3.0  # Arrow length
            marker.scale.y = 0.8  # Arrow width
            marker.scale.z = 0.8  # Arrow height
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = data['color']

            # Determine the position to display the marker
            if data['assigned_point']:
                # self.get_logger().info(f"{robot_name}: used assigned point")
                position = data['assigned_point']
                marker.pose.position.x = float(position.x)
                marker.pose.position.y = float(position.y)
                marker.pose.position.z = 0.0
            # elif data['current_point']:
            #     self.get_logger().info(f"{robot_name}: used current point")
            #     position = data['current_point']
            #     marker.pose.position.x = float(position.x)
            #     marker.pose.position.y = float(position.y)
            #     marker.pose.position.z = 0.0
            else:
                # self.get_logger().info(f"{robot_name}: used origin")
                position = "origin"  # If no position available
                marker.pose.position.x = 0.0
                marker.pose.position.y = 0.0
                marker.pose.position.z = 0.0

            # Find the nearest object to the robot
            nearest_object_position = self.get_nearest_object(position)

            if nearest_object_position and position != "origin":
                dx = nearest_object_position.x - marker.pose.position.x
                dy = nearest_object_position.y - marker.pose.position.y
                yaw = math.atan2(dy, dx)  # Angle in radians

                # Convert the angle to a quaternion for orientation
                q = quaternion_from_euler(0.0, 0.0, yaw)  # Only yaw (rotation around Z axis)
                marker.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
                # self.get_logger().info(f"Set yaw for {robot_name}: {yaw} radians")
            else:
                marker.pose.orientation.w = 1.0

            # Add the marker to this robot's MarkerArray
            marker_array.markers.append(marker)

            # Publish the MarkerArray using this robot's specific publisher
            data['marker_publisher_point'].publish(marker_array)
            # self.get_logger().info(f"Published marker for {robot_name}")

    def get_nearest_object(self, current_position):
        """
        Returns the nearest object based on the robot's current position and a list of object positions.
        """
        objects_positions = self.calculate_object_positions()

        if current_position == "origin" or not objects_positions:
            return None  # No valid position or no objects to compare

        nearest_distance = float('inf')
        nearest_object = None
        
        for obj in objects_positions:
            distance = math.sqrt((current_position.x - obj.x)**2 + (current_position.y - obj.y)**2)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_object = obj

        return nearest_object

    def calculate_object_positions(self):
        """
        Process the occupancy grid in self.occupancy_grid to calculate the positions
        of occupied cells (objects) in the world frame.

        :return: List of geometry_msgs.msg.Point objects representing object positions.
        """
        object_positions = []  # List to store object positions

        # Extract grid metadata
        width = self.width
        height = self.height
        resolution = self.resolution
        origin = self.origin  # Grid's origin in world coordinates
        data = self.occupancy_grid  # Assuming occupancy_grid is a 2D numpy array

        # Find all occupied cells
        occupied_indices = np.argwhere(data == 1)

        # Convert grid points to world coordinates using the transform
        transformed_points = self.convert_grid_to_world(
            Atransform(occupied_indices, data.shape, self.get_logger())
        )
        for x, y in transformed_points:
            object_positions.append(Point(x=x, y=y, z=0.0))

        # Log the number of objects detected (optional)
        # self.get_logger().info(f"Detected {len(object_positions)} objects.")

        return object_positions


def main(args=None):
    rclpy.init(args=args)
    node = GoalAssignerNode()
    try:
        # Use MultiThreadedExecutor to allow callbacks to run in parallel
        executor = MultiThreadedExecutor(num_threads=20)
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()




