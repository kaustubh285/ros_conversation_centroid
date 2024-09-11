#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

class ConversationDetector(Node):

    def __init__(self):
        super().__init__("conversation_detector_node")
        self.get_logger().info("Hello")



def main(args=None):
    rclpy.init(args=args)

    node = ConversationDetector()
    rclpy.spin(node)

    rclpy.shutdown()


if __name__=="__main__":
    main()