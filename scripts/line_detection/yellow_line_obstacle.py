#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField


class YellowLineObstacle(Node):

    def __init__(self):
        super().__init__('yellow_line_obstacle')

        self.sub = self.create_subscription(
            PointCloud2,
            '/line_detector/yellow',
            self.pointcloud_callback,
            10
        )
        self.pub = self.create_publisher(
            PointCloud2,
            '/line_detector/yellow_obstacle',
            10
        )

    def pointcloud_callback(self, msg: PointCloud2):
        if msg.width == 0:
            return

        # Each point is 4x float32: x, y, z, rgb — just grab xyz
        buf = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 4)
        if len(buf) == 0:
            return

        xyz = buf[:, :3].copy()

        # Inflate vertically so the costmap obstacle layer sees 3D points
        offsets = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
        inflated = np.concatenate(
            [xyz + np.array([[0, 0, dz]], dtype=np.float32) for dz in offsets],
            axis=0
        )

        self.pub.publish(self._make_pointcloud2(inflated, msg.header.stamp))

    def _make_pointcloud2(self, xyz, stamp):
        fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]

        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = 'map'
        msg.height = 1
        msg.width = len(xyz)
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * len(xyz)
        msg.data = xyz.astype(np.float32).tobytes()
        msg.is_dense = True
        return msg


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(YellowLineObstacle())


if __name__ == '__main__':
    main()