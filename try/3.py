import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from custom_msgs.msg import ObjectInfo  # Assuming custom message type
import sqlite3
import time


class ObjectTracker(Node):
    def __init__(self):
        super().__init__('object_tracker')

        # Subscribe to the topic where robots publish object data
        self.subscription = self.create_subscription(
            ObjectInfo,
            '/robot_object_info',
            self.object_callback,
            10
        )

        # Publisher for MarkerArray to visualize objects
        self.marker_publisher = self.create_publisher(MarkerArray, '/object_markers', 10)

        # Initialize SQLite database
        self.init_database()

        # Timer to periodically publish markers
        self.timer = self.create_timer(1.0, self.publish_markers)

        self.marker_array = MarkerArray()

    def init_database(self):
        # Connect to SQLite database
        self.conn = sqlite3.connect('object_data.db')
        self.cursor = self.conn.cursor()

        # Create table to store object information
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS objects (
                localObjectId INTEGER PRIMARY KEY,
                objectClass TEXT,
                objectPosX REAL,
                objectPosY REAL,
                objectPosZ REAL,
                last_seen REAL
            )
        ''')
        self.conn.commit()

    def object_callback(self, msg):
        # Update or insert the object information into the database
        self.cursor.execute('''
            INSERT INTO objects (localObjectId, objectClass, objectPosX, objectPosY, objectPosZ, last_seen)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(localObjectId) DO UPDATE SET
                objectClass=excluded.objectClass,
                objectPosX=excluded.objectPosX,
                objectPosY=excluded.objectPosY,
                objectPosZ=excluded.objectPosZ,
                last_seen=excluded.last_seen
        ''', (msg.localObjectId, msg.objectClass, msg.objectPosX, msg.objectPosY, msg.objectPosZ, time.time()))
        self.conn.commit()

    def publish_markers(self):
        # Clear the current marker array
        self.marker_array.markers.clear()

        # Fetch objects from the database
        self.cursor.execute('SELECT localObjectId, objectClass, objectPosX, objectPosY, objectPosZ FROM objects')
        rows = self.cursor.fetchall()

        for row in rows:
            localObjectId, objectClass, x, y, z = row
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "objects"
            marker.id = localObjectId
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = z
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.5
            marker.color.a = 1.0
            marker.color.r = 1.0  # Red for now
            marker.color.g = 0.0
            marker.color.b = 0.0
            self.marker_array.markers.append(marker)

        # Publish the marker array
        self.marker_publisher.publish(self.marker_array)

    def shutdown(self):
        # Close database connection on shutdown
        self.conn.close()


def main(args=None):
    rclpy.init(args=args)
    object_tracker = ObjectTracker()

    try:
        rclpy.spin(object_tracker)
    except KeyboardInterrupt:
        pass
    finally:
        object_tracker.shutdown()
        object_tracker.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
