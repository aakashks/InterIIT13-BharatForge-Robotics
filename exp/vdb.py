import rclpy
from rclpy.node import Node
from pymongo import MongoClient
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import json


class CentralNode(Node):
    def __init__(self):
        super().__init__('central_node')
        self.subscription = self.create_subscription(
            String,
            'object_info',
            self.listener_callback,
            10
        )
        self.marker_publisher = self.create_publisher(MarkerArray, 'visualization_marker_array', 10)
        self.marker_array = MarkerArray()
        self.db_client = MongoClient('mongodb://localhost:27017/')  # Connect to MongoDB
        self.database = self.db_client['object_database']          # Database name
        self.collection = self.database['objects']                # Collection name
        self.sno = 0  # Unique ID for markers

    def listener_callback(self, msg):
        try:
            # Parse JSON data from the message
            data = json.loads(msg.data)
            local_object_id = data['localObjectId']
            object_class = data['objectClass']
            object_pos_x = data['objectPosX']
            object_pos_y = data['objectPosY']
            
            # Insert or update in MongoDB
            self.collection.update_one(
                {'localObjectId': local_object_id},
                {'$set': {
                    'objectClass': object_class,
                    'objectPosX': object_pos_x,
                    'objectPosY': object_pos_y
                }},
                upsert=True
            )
            
            self.get_logger().info(f"Received and stored object: {data}")
            
            # Add Marker for RViz visualization
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "poses"
            marker.id = self.sno
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = object_pos_x
            marker.pose.position.y = object_pos_y
            marker.pose.position.z = 0.0  # Assuming objects are on the ground
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.5
            marker.color.a = 1.0
            marker.color.r = 1.0  # Red color for markers
            marker.color.g = 0.0
            marker.color.b = 0.0
            
            self.marker_array.markers.append(marker)
            self.sno += 1
            
            # Publish Marker Array
            self.marker_publisher.publish(self.marker_array)

        except Exception as e:
            self.get_logger().error(f"Error processing message: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = CentralNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down central node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()