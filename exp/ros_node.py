## Template code for ROS2 connector node
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class ExampleNode(Node):
    def __init__(self):
        super().__init__('example_node')
        self.publisher_ = self.create_publisher(String, 'ros_feedback', 10)
        self.subscription = self.create_subscription(
            String,
            'coordinate_data',
            self.listener_callback,
            10
        )

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: {msg.data}')

    def publish_message(self, message):
        msg = String()
        msg.data = message
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ExampleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
