#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time

from nav2_msgs.action import NavigateToPose

from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray


from sensor_msgs_py import point_cloud2 as pc2

import tf2_ros



class LineMapper(Node):

    def __init__(self):
        super().__init__('line_mapper')


        qos_be = QoSProfile(depth=5,
                            reliability=ReliabilityPolicy.BEST_EFFORT,
                            history=HistoryPolicy.KEEP_LAST)

        self.create_subscription(
            PointCloud2, '/line_detector/blue', self._cloud_cb, qos_be)


        self._nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self._tf_buf = tf2_ros.Buffer()
        self._tf_ls  = tf2_ros.TransformListener(self._tf_buf, self)

        # Publisher for the skeleton visualization (rviz)
        self._skeleton_pub = self.create_publisher(MarkerArray, '/line_mapper/skeleton', 10)

        # parameters
        self._cluster_dist = 0.05  # meters: distance threshold to connect nearby points into same component
        self._min_branch_len = 0.04  # meters: ignore leaf branches shorter than this
        self._max_points_for_processing = 1000


    def _cloud_cb(self, cloud: PointCloud2):
        # Read points from PointCloud2 into NxM numpy array (skips NaNs)
        # read points and coerce to plain python float tuples to avoid structured numpy dtypes
        points = [ (float(p[0]), float(p[1]), float(p[2]))
                   for p in pc2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True) ]
        if not points:
            return

        # stack into 2D float array
        pts = np.array(points, dtype=float)

        # work in XY plane for skeleton
        xy = pts[:, :2].astype(float)

        # downsample if too many points
        n = xy.shape[0]
        if n > self._max_points_for_processing:
            idx = np.random.choice(n, self._max_points_for_processing, replace=False)
            xy = xy[idx]
            n = xy.shape[0]

        # Build adjacency by proximity to find connected components (lines/junctions)
        dists = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
        adj = dists < self._cluster_dist

        # connected components via BFS
        visited = np.zeros(n, dtype=bool)
        components = []
        for i in range(n):
            if visited[i]:
                continue
            stack = [i]
            comp = []
            visited[i] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                neigh = np.where(adj[u])[0]
                for v in neigh:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            components.append(np.array(comp, dtype=int))

        if len(components) == 0:
            return

        # Determine robot origin in the pointcloud frame. Prefer TF lookup, fallback to origin.
        robot_xy = np.array([0.0, 0.0])
        try:
            # attempt to get transform of base_link in cloud frame
            tf = self._tf_buf.lookup_transform(cloud.header.frame_id, 'base_link', Time())
            robot_xy = np.array([tf.transform.translation.x, tf.transform.translation.y])
        except Exception:
            # couldn't lookup TF -- assume robot at origin of cloud frame
            pass

        # If multiple disconnected components, pick the one whose centroid is nearest to robot
        if len(components) > 1:
            centroids = [xy[c].mean(axis=0) for c in components]
            d_to_robot = [np.linalg.norm(c - robot_xy) for c in centroids]
            chosen_idx = int(np.argmin(d_to_robot))
            comp_idx = components[chosen_idx]
        else:
            comp_idx = components[0]

        comp_pts = xy[comp_idx]
        m = comp_pts.shape[0]
        if m == 0:
            return
        if m == 1:
            # single point: publish a single sphere marker
            marker_array = MarkerArray()
            sph = Marker()
            sph.header.frame_id = cloud.header.frame_id
            sph.header.stamp = cloud.header.stamp
            sph.ns = 'skeleton_nodes'
            sph.id = 0
            sph.type = Marker.SPHERE
            sph.action = Marker.ADD
            sph.pose.position.x = float(comp_pts[0, 0])
            sph.pose.position.y = float(comp_pts[0, 1])
            sph.pose.position.z = 0.0
            sph.scale.x = 0.06
            sph.scale.y = 0.06
            sph.scale.z = 0.06
            sph.color.r = 0.0
            sph.color.g = 1.0
            sph.color.b = 0.0
            sph.color.a = 1.0
            marker_array.markers.append(sph)
            self._skeleton_pub.publish(marker_array)
            return

        # Build a minimum spanning tree (MST) over the component points to form a connected skeleton
        # Prim's algorithm (O(m^2) fine for reasonable m)
        comp_dists = np.linalg.norm(comp_pts[:, None, :] - comp_pts[None, :, :], axis=2)
        in_mst = np.zeros(m, dtype=bool)
        in_mst[0] = True
        nearest = np.full(m, 0, dtype=int)
        key = comp_dists[0].copy()
        edges = []
        for _ in range(m - 1):
            # select next vertex to add
            key[in_mst] = np.inf
            v = int(np.argmin(key))
            u = int(nearest[v])
            edges.append((u, v))
            in_mst[v] = True
            # update keys
            for w in range(m):
                if not in_mst[w] and comp_dists[v, w] < key[w]:
                    key[w] = comp_dists[v, w]
                    nearest[w] = v

        # Remove very short leaf branches before classifying junctions.
        # This trims tiny stubs that would otherwise show up as false junctions.
        adjacency = [set() for _ in range(m)]
        for u, v in edges:
            adjacency[u].add(v)
            adjacency[v].add(u)

        changed = True
        while changed:
            changed = False
            degs = [len(neigh) for neigh in adjacency]
            leaf_nodes = [i for i, deg in enumerate(degs) if deg == 1]

            for leaf in leaf_nodes:
                if len(adjacency[leaf]) != 1:
                    continue

                path = [leaf]
                prev = -1
                current = leaf
                branch_len = 0.0
                terminal = leaf

                while True:
                    neighbors = [n for n in adjacency[current] if n != prev]
                    if not neighbors:
                        break

                    nxt = neighbors[0]
                    branch_len += float(comp_dists[current, nxt])
                    path.append(nxt)
                    prev, current = current, nxt

                    if len(adjacency[current]) != 2:
                        terminal = current
                        break

                if terminal != leaf and len(adjacency[terminal]) >= 3 and branch_len < self._min_branch_len:
                    for a, b in zip(path[:-1], path[1:]):
                        if b in adjacency[a]:
                            adjacency[a].remove(b)
                            adjacency[b].remove(a)
                            changed = True

        edges = []
        for u in range(m):
            for v in adjacency[u]:
                if u < v:
                    edges.append((u, v))

        # Prepare MarkerArray: one LINE_LIST marker for edges, plus optional spheres at junctions
        marker_array = MarkerArray()

        # LINE_LIST marker
        line_marker = Marker()
        line_marker.header.frame_id = cloud.header.frame_id
        line_marker.header.stamp = cloud.header.stamp
        line_marker.ns = 'skeleton'
        line_marker.id = 0
        line_marker.type = Marker.LINE_LIST
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.02
        line_marker.color.r = 0.0
        line_marker.color.g = 0.0
        line_marker.color.b = 1.0
        line_marker.color.a = 1.0

        # add edge pairs
        for (u, v) in edges:
            p_u = Point(x=float(comp_pts[u, 0]), y=float(comp_pts[u, 1]), z=0.0)
            p_v = Point(x=float(comp_pts[v, 0]), y=float(comp_pts[v, 1]), z=0.0)
            line_marker.points.append(p_u)
            line_marker.points.append(p_v)

        marker_array.markers.append(line_marker)

        # compute degrees to find junctions
        deg = np.zeros(m, dtype=int)
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1

        # Add small spheres for junctions (degree > 2) and endpoints (degree == 1)
        marker_id = 1
        for i in range(m):
            if deg[i] > 2 or deg[i] == 1:
                sph = Marker()
                sph.header.frame_id = cloud.header.frame_id
                sph.header.stamp = cloud.header.stamp
                sph.ns = 'skeleton_nodes'
                sph.id = marker_id
                sph.type = Marker.SPHERE
                sph.action = Marker.ADD
                sph.pose.position.x = float(comp_pts[i, 0])
                sph.pose.position.y = float(comp_pts[i, 1])
                sph.pose.position.z = 0.0
                sph.scale.x = 0.06
                sph.scale.y = 0.06
                sph.scale.z = 0.06
                if deg[i] > 2:
                    # junction: red
                    sph.color.r = 1.0
                    sph.color.g = 0.0
                    sph.color.b = 0.0
                    sph.color.a = 1.0
                else:
                    # endpoint: green
                    sph.color.r = 0.0
                    sph.color.g = 1.0
                    sph.color.b = 0.0
                    sph.color.a = 1.0
                marker_array.markers.append(sph)
                marker_id += 1

        # Publish MarkerArray
        self._skeleton_pub.publish(marker_array)

# ============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = LineMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()