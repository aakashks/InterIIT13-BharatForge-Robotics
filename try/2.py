import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import random
import time


class RobotNode(Node):
    def __init__(self):
        super().__init__('robot_node')
        self.publisher = self.create_publisher(String, 'object_info', 10)
        self.timer = self.create_timer(1.0, self.publish_object_info)

    def publish_object_info(self):
        # Simulating object detection
        if random.random() < 0.5:  # 50% chance of detecting an object
            data = {
                'localObjectId': random.randint(1, 1000),
                'objectClass': 'flower pot',
                'objectPosX': random.uniform(-10, 10),
                'objectPosY': random.uniform(-10, 10)
            }
            msg = String()
            msg.data = json.dumps(data)
            self.publisher.publish(msg)
            self.get_logger().info(f"Published: {data}")


def main(args=None):
    rclpy.init(args=args)
    node = RobotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down robot node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()