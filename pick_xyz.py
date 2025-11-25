#!/usr/bin/env python3
import time
import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK, GetPositionIK_Request
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory


class IKPick(Node):
    def __init__(self):
        super().__init__("ik_pick")

        # IK service client (MoveIt2)
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")

        # Arm trajectory action client
        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/left_arm_controller/follow_joint_trajectory",
        )

        # Gripper trajectory action client
        self.gripper_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/left_gripper_controller/follow_joint_trajectory",
        )

        # Joint order used by controller
        self.joint_names = [
            "left_panda_joint1",
            "left_panda_joint2",
            "left_panda_joint3",
            "left_panda_joint4",
            "left_panda_joint5",
            "left_panda_joint6",
            "left_panda_joint7",
        ]

        # Initial "steady/ready" state from SRDF
        self.ready_joints = [
            0.0,        # left_panda_joint1
            -0.785,     # left_panda_joint2
            0.0,        # left_panda_joint3
            -2.356,     # left_panda_joint4
            0.0,        # left_panda_joint5
            1.571,      # left_panda_joint6
            0.785,      # left_panda_joint7
        ]

        # Track current joint estimate (for better IK seed)
        self.current_joints = list(self.ready_joints)

    # ------------------------------
    # Helpers
    # ------------------------------
    def wait_for_servers(self):
        # Wait for IK
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /compute_ik service...")

        # Wait for controllers
        self.get_logger().info("Waiting for arm controller action server...")
        self.arm_client.wait_for_server()
        self.get_logger().info("Waiting for gripper controller action server...")
        self.gripper_client.wait_for_server()

    def compute_ik(self, x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
        """
        Call MoveIt2 IK to get joint angles for target (x, y, z, rpy).

        roll, pitch, yaw are in radians, in the 'world' frame.
        """
        req = GetPositionIK_Request()
        req.ik_request.group_name = "left_panda_arm"

        pose = PoseStamped()
        pose.header.frame_id = "world"  # adjust if your planning frame is different
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)

        # --- RPY -> quaternion ---
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        req.ik_request.pose_stamped = pose

        # Use current joints as IK seed
        req.ik_request.robot_state.joint_state.name = self.joint_names
        req.ik_request.robot_state.joint_state.position = list(self.current_joints)

        future = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()

        if res is None:
            raise RuntimeError("No IK response")

        if res.error_code.val != res.error_code.SUCCESS:
            self.get_logger().error(f"IK failed with code {res.error_code.val}")
            raise RuntimeError("IK failed")

        js = res.solution.joint_state
        pos_map = {name: pos for name, pos in zip(js.name, js.position)}
        q = [float(pos_map[j]) for j in self.joint_names]
        return q

    def send_arm_trajectory(self, positions, duration_sec=2.0):
        traj = JointTrajectory()
        traj.joint_names = list(self.joint_names)

        pt = JointTrajectoryPoint()
        pt.positions = list(positions)
        pt.time_from_start.sec = int(duration_sec)
        pt.time_from_start.nanosec = 0
        traj.points.append(pt)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        send_future = self.arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Arm goal rejected")
            return

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        # Update our current joint estimate
        self.current_joints = list(positions)

        self.get_logger().info("Arm trajectory finished")

    def send_gripper(self, opening, duration_sec=1.0):
        """opening in [0.0, ~0.04] for Panda."""
        traj = JointTrajectory()
        traj.joint_names = ["left_panda_finger_joint1"]

        pt = JointTrajectoryPoint()
        pt.positions = [float(opening)]
        pt.time_from_start.sec = int(duration_sec)
        pt.time_from_start.nanosec = 0
        traj.points.append(pt)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        send_future = self.gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Gripper goal rejected")
            return

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info("Gripper motion finished")

    # ------------------------------
    # Main sequence
    # ------------------------------
    def run(self):
        self.wait_for_servers()

        # -------------------- 0) Move to READY pose --------------------
        self.get_logger().info("Moving to initial READY steady state")
        self.send_arm_trajectory(self.ready_joints, duration_sec=3.0)
        time.sleep(1.0)

        self.get_logger().info("Opening gripper to initial state")
        self.send_gripper(0.04, duration_sec=1.0)
        time.sleep(1.0)

        # We want the tool to point DOWN toward the ground when near the object.
        # Assuming default tool frame points along +X, then pitch = -pi/2 tilts it down.
        down_pitch = -math.pi / 2.0
        down_roll = 0.0
        down_yaw = 0.0   # adjust if you need to rotate around vertical axis as well

        # -------------------- 1) Above object (0,0,0.20) --------------------
        self.get_logger().info("Moving above object (0,0,0.20) with tool pointing down")
        q_above = self.compute_ik(
            0.0, 0.0, 0.20,
            roll=down_roll,
            pitch=down_pitch,
            yaw=down_yaw,
        )
        self.send_arm_trajectory(q_above, duration_sec=3.0)
        time.sleep(1.0)

        # -------------------- 2) Open gripper --------------------
        self.get_logger().info("Opening gripper before descend")
        self.send_gripper(0.04, duration_sec=1.0)
        time.sleep(1.0)

        # -------------------- 3) Move down near (0,0,0) --------------------
        self.get_logger().info("Moving down to near (0,0,0) with tool pointing down")
        q_down = self.compute_ik(
            0.0, 0.0, 0.05,   # adjust z if table height is different
            roll=down_roll,
            pitch=down_pitch,
            yaw=down_yaw,
        )
        self.send_arm_trajectory(q_down, duration_sec=2.0)
        time.sleep(1.0)

        # -------------------- 4) Close gripper --------------------
        self.get_logger().info("Closing gripper to grasp")
        self.send_gripper(0.0, duration_sec=1.0)
        time.sleep(1.0)

        # -------------------- 5) Lift object --------------------
        self.get_logger().info("Lifting object with tool still pointing down")
        q_lift = self.compute_ik(
            0.0, 0.0, 0.25,
            roll=down_roll,
            pitch=down_pitch,
            yaw=down_yaw,
        )
        self.send_arm_trajectory(q_lift, duration_sec=3.0)
        self.get_logger().info("DONE")


def main():
    rclpy.init()
    node = IKPick()
    node.run()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
