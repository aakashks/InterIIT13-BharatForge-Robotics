import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header, String
from geometry_msgs.msg import Point
import json
from datetime import datetime

class ObjectTracker(Node):
    def __init__(self):
        super().__init__('object_tracker')
        
        # Initialize JSON file for storing object detections
        self.json_file = 'detected_objects.json'
        self.load_data()
        
        # Create a publisher for visualization markers
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            'visualization_marker_array',
            10
        )
        
        # Subscribe to object detection topics from all robots
        # Assuming each robot publishes to 'robot_X/object_detection'
        self.object_subscriptions = []
        for robot_id in range(1, 5):  # Adjust range based on number of robots
            topic = f'robot_{robot_id}/object_detection'
            sub = self.create_subscription(
                String,
                topic,
                self.object_detection_callback,
                10
            )
            self.object_subscriptions.append(sub)
        
        # Timer for periodic visualization update
        self.create_timer(1.0, self.publish_visualization)
        
        self.get_logger().info('Object Tracker initialized')

    def load_data(self):
        try:
            with open(self.json_file, 'r') as file:
                self.data = json.load(file)
        except FileNotFoundError:
            self.data = {}

    def save_data(self):
        with open(self.json_file, 'w') as file:
            json.dump(self.data, file, default=str)

    def object_detection_callback(self, msg):
        # Parse JSON string message
        detected_objects = json.loads(msg.data)
        
        for obj in detected_objects:
            # Store object detection in JSON file
            object_data = {
                'local_object_id': obj['id'],
                'object_class': obj['label'],
                'position': obj['position'],
                'timestamp': datetime.now().isoformat(),
                'robot_id': obj['robot_id']
            }
            
            # Update or insert the object data
            self.data[obj['id']] = object_data
            self.save_data()
            
            self.get_logger().info(f'Received object detection from robot {obj["robot_id"]}')

    def publish_visualization(self):
        marker_array = MarkerArray()
        
        # Retrieve all objects from JSON file
        objects = self.data.values()
        
        for obj in objects:
            marker = Marker()
            marker.header = Header()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            
            marker.ns = "objects"
            marker.id = obj['local_object_id']
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # Set position
            marker.pose.position = Point()
            marker.pose.position.x = obj['position']['x']
            marker.pose.position.y = obj['position']['y']
            marker.pose.position.z = obj['position']['z']
            
            # Set scale
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.5
            
            # Set color based on object class (you can customize this)
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            
            marker_array.markers.append(marker)
        
        # Publish marker array
        self.marker_publisher.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectTracker()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()