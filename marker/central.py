import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from custom_interfaces.msg import ObjectDetection  # You'll need to create this custom message
from pymongo import MongoClient
from datetime import datetime

class ObjectTracker(Node):
    def __init__(self):
        super().__init__('object_tracker')
        
        # Initialize MongoDB connection
        self.client = MongoClient('mongodb://localhost:27017/')
        self.db = self.client['object_tracking']
        self.collection = self.db['detected_objects']
        
        # Create a publisher for visualization markers
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            'visualization_marker_array',
            10
        )
        
        # Subscribe to object detection topics from all robots
        # Assuming each robot publishes to 'robot_X/object_detection'
        self.subscriptions = []
        for robot_id in range(100):  # Adjust range based on number of robots
            topic = f'robot_{robot_id}/object_detection'
            sub = self.create_subscription(
                ObjectDetection,
                topic,
                self.object_detection_callback,
                10
            )
            self.subscriptions.append(sub)
        
        # Timer for periodic visualization update
        self.create_timer(1.0, self.publish_visualization)
        
        self.get_logger().info('Object Tracker initialized')

    def object_detection_callback(self, msg):
        # Store object detection in MongoDB
        object_data = {
            'local_object_id': msg.local_object_id,
            'object_class': msg.object_class,
            'position': {
                'x': msg.position.x,
                'y': msg.position.y,
                'z': msg.position.z
            },
            'timestamp': datetime.now(),
            'robot_id': msg.robot_id
        }
        
        # Update or insert the object data
        self.collection.update_one(
            {'local_object_id': msg.local_object_id},
            {'$set': object_data},
            upsert=True
        )
        
        self.get_logger().info(f'Received object detection from robot {msg.robot_id}')

    def publish_visualization(self):
        marker_array = MarkerArray()
        
        # Retrieve all objects from MongoDB
        objects = self.collection.find()
        
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