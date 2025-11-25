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


class DualPickPlace(Node):
    def __init__(self):
        super().__init__("dual_pick_place")

        # ===== IK client =====
        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")

        # ===== Arm action clients =====
        self.left_arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/left_arm_controller/follow_joint_trajectory",
        )
        self.right_arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/right_arm_controller/follow_joint_trajectory",
        )

        # ===== Gripper action clients =====
        self.left_gripper_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/left_gripper_controller/follow_joint_trajectory",
        )
        self.right_gripper_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/right_gripper_controller/follow_joint_trajectory",
        )

        # Joint names
        self.left_joint_names = [
            "left_panda_joint1",
            "left_panda_joint2",
            "left_panda_joint3",
            "left_panda_joint4",
            "left_panda_joint5",
            "left_panda_joint6",
            "left_panda_joint7",
        ]
        self.right_joint_names = [
            "right_panda_joint1",
            "right_panda_joint2",
            "right_panda_joint3",
            "right_panda_joint4",
            "right_panda_joint5",
            "right_panda_joint6",
            "right_panda_joint7",
        ]

        # READY poses (adjust to your SRDF if needed)
        self.left_ready = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        self.right_ready = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

        self.left_current = list(self.left_ready)
        self.right_current = list(self.right_ready)

        # ===== Orientations =====
        # Left: gripper pointing UP (toward sky)
        self.left_roll = 0.0
        self.left_pitch = +math.pi / 2.0
        self.left_yaw = 0.0

        # Right: gripper pointing DOWN (toward ground)
        self.right_roll = 0.0
        self.right_pitch = -math.pi / 2.0
        self.right_yaw = 0.0

        # ===== Positions (tune to your scene) =====
        # Start object pose for LEFT (Phase 1)
        self.start_x = 0.40
        self.start_y = 0.15
        self.start_z = 0.00    # table height

        # Middle / transfer pose (object is placed here by LEFT, picked by RIGHT)
        self.mid_x = 0.35
        self.mid_y = 0.00
        self.mid_z = 0.00      # table height

        # Final place pose for RIGHT
        self.final_x = 0.30
        self.final_y = -0.20
        self.final_z = 0.00    # table height

        # ARC waypoint for RIGHT to swing around (counterclockwise / outside)
        # Adjust these if they still pass too close to the left arm.
        self.right_arc_x = 0.45
        self.right_arc_y = -0.05
        self.right_arc_z = 0.00  # table height

    # ======================================================
    #   Helpers
    # ======================================================
    def wait_for_servers(self):
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /compute_ik...")

        self.get_logger().info("Waiting for left arm controller...")
        self.left_arm_client.wait_for_server()
        self.get_logger().info("Waiting for right arm controller...")
        self.right_arm_client.wait_for_server()
        self.get_logger().info("Waiting for left gripper controller...")
        self.left_gripper_client.wait_for_server()
        self.get_logger().info("Waiting for right gripper controller...")
        self.right_gripper_client.wait_for_server()

    def rpy_to_quat(self, r, p, y):
        cy = math.cos(y * 0.5)
        sy = math.sin(y * 0.5)
        cp = math.cos(p * 0.5)
        sp = math.sin(p * 0.5)
        cr = math.cos(r * 0.5)
        sr = math.sin(r * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return qx, qy, qz, qw

    def compute_ik(
        self,
        group_name,
        joint_names,
        current_joints,
        x,
        y,
        z,
        roll,
        pitch,
        yaw,
    ):
        req = GetPositionIK_Request()
        req.ik_request.group_name = group_name

        pose = PoseStamped()
        pose.header.frame_id = "world"   # check this matches MoveIt planning frame
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)

        qx, qy, qz, qw = self.rpy_to_quat(roll, pitch, yaw)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        req.ik_request.pose_stamped = pose
        req.ik_request.robot_state.joint_state.name = joint_names
        req.ik_request.robot_state.joint_state.position = list(current_joints)
        req.ik_request.avoid_collisions = False  # set True later if geometry is clean

        future = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()

        if res is None:
            raise RuntimeError("No IK response")

        if res.error_code.val != res.error_code.SUCCESS:
            self.get_logger().error(
                f"IK failed for {group_name}, code {res.error_code.val} "
                f"at x={x:.3f}, y={y:.3f}, z={z:.3f}"
            )
            raise RuntimeError("IK failed")

        js = res.solution.joint_state
        pos_map = {n: p for n, p in zip(js.name, js.position)}
        return [float(pos_map[j]) for j in joint_names]

    def send_arm_trajectory(self, client, joint_names, current_ref, positions, duration):
        traj = JointTrajectory()
        traj.joint_names = list(joint_names)

        pt = JointTrajectoryPoint()
        pt.positions = list(positions)
        pt.time_from_start.sec = int(duration)
        pt.time_from_start.nanosec = 0
        traj.points.append(pt)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        send_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        gh = send_future.result()

        if not gh.accepted:
            self.get_logger().error("Arm goal rejected")
            return False

        res_future = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_future)

        for i in range(len(current_ref)):
            current_ref[i] = positions[i]

        self.get_logger().info("Arm trajectory finished")
        return True

    def send_gripper(self, client, opening, left=True, duration=1.0):
        traj = JointTrajectory()
        traj.joint_names = [
            "left_panda_finger_joint1" if left else "right_panda_finger_joint1"
        ]

        pt = JointTrajectoryPoint()
        pt.positions = [float(opening)]
        pt.time_from_start.sec = int(duration)
        pt.time_from_start.nanosec = 0
        traj.points.append(pt)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        send_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        gh = send_future.result()
        if not gh.accepted:
            self.get_logger().error("Gripper goal rejected")
            return False

        res_future = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_future)
        self.get_logger().info("Gripper motion finished")
        return True

    # ======================================================
    #   Main sequence
    # ======================================================
    def run(self):
        self.wait_for_servers()

        # ---------- PHASE 1: LEFT ONLY ----------
        # Right arm does not move at all here.

        # Ensure left is in READY
        self.get_logger().info("LEFT PHASE: move LEFT to READY...")
        self.send_arm_trajectory(
            self.left_arm_client,
            self.left_joint_names,
            self.left_current,
            self.left_ready,
            duration=3.0,
        )
        self.get_logger().info("RIGHT PHASE: RIGHT back to READY (steady)")
        self.send_arm_trajectory(
            self.right_arm_client,
            self.right_joint_names,
            self.right_current,
            self.right_ready,
            duration=3.0,
        )
        time.sleep(0.5)

        # Open left gripper
        self.send_gripper(self.left_gripper_client, 0.04, left=True)
        time.sleep(0.5)

        # a) LEFT: pick object from start pose (tool UP)
        z_start_above = self.start_z + 0.20
        z_start_pick = self.start_z + 0.03

        self.get_logger().info("LEFT PHASE: LEFT above start (UP)")
        q_left_start_above = self.compute_ik(
            "left_panda_arm",
            self.left_joint_names,
            self.left_current,
            self.start_x,
            self.start_y,
            z_start_above,
            self.left_roll,
            self.left_pitch,
            self.left_yaw,
        )
        self.send_arm_trajectory(
            self.left_arm_client,
            self.left_joint_names,
            self.left_current,
            q_left_start_above,
            duration=3.0,
        )
        time.sleep(0.5)

        self.get_logger().info("LEFT PHASE: LEFT down to start (UP)")
        q_left_start_pick = self.compute_ik(
            "left_panda_arm",
            self.left_joint_names,
            self.left_current,
            self.start_x,
            self.start_y,
            z_start_pick,
            self.left_roll,
            self.left_pitch,
            self.left_yaw,
        )
        self.send_arm_trajectory(
            self.left_arm_client,
            self.left_joint_names,
            self.left_current,
            q_left_start_pick,
            duration=2.0,
        )
        time.sleep(0.5)

        self.get_logger().info("LEFT PHASE: LEFT close gripper at start")
        self.send_gripper(self.left_gripper_client, 0.0, left=True)
        time.sleep(0.5)

        self.get_logger().info("LEFT PHASE: LEFT lift from start (UP)")
        self.send_arm_trajectory(
            self.left_arm_client,
            self.left_joint_names,
            self.left_current,
            q_left_start_above,
            duration=3.0,
        )
        time.sleep(0.5)

        # b) LEFT: move to middle and drop (tool UP)
        z_mid_above = self.mid_z + 0.20
        z_mid_place = self.mid_z + 0.03

        self.get_logger().info("LEFT PHASE: LEFT above middle (UP)")
        q_left_mid_above = self.compute_ik(
            "left_panda_arm",
            self.left_joint_names,
            self.left_current,
            self.mid_x,
            self.mid_y,
            z_mid_above,
            self.left_roll,
            self.left_pitch,
            self.left_yaw,
        )
        self.send_arm_trajectory(
            self.left_arm_client,
            self.left_joint_names,
            self.left_current,
            q_left_mid_above,
            duration=3.0,
        )
        time.sleep(0.5)

        self.get_logger().info("LEFT PHASE: LEFT down to middle (UP)")
        q_left_mid_place = self.compute_ik(
            "left_panda_arm",
            self.left_joint_names,
            self.left_current,
            self.mid_x,
            self.mid_y,
            z_mid_place,
            self.left_roll,
            self.left_pitch,
            self.left_yaw,
        )
        self.send_arm_trajectory(
            self.left_arm_client,
            self.left_joint_names,
            self.left_current,
            q_left_mid_place,
            duration=2.0,
        )
        time.sleep(0.5)

        self.get_logger().info("LEFT PHASE: LEFT open gripper at middle")
        self.send_gripper(self.left_gripper_client, 0.04, left=True)
        time.sleep(0.5)

        self.get_logger().info("LEFT PHASE: LEFT lift from middle (UP)")
        self.send_arm_trajectory(
            self.left_arm_client,
            self.left_joint_names,
            self.left_current,
            q_left_mid_above,
            duration=3.0,
        )
        time.sleep(0.5)

        # c) LEFT: go back to READY (steady)
        self.get_logger().info("LEFT PHASE: LEFT back to READY (steady)")
        self.send_arm_trajectory(
            self.left_arm_client,
            self.left_joint_names,
            self.left_current,
            self.left_ready,
            duration=3.0,
        )
        time.sleep(1.0)

        # At this point:
        #   - LEFT is steady in READY
        #   - Object is at middle pose
        #   - RIGHT has not moved

        # ---------- PHASE 2: RIGHT ONLY ----------
        # LEFT stays steady; RIGHT does middle -> arc -> final

        self.get_logger().info("RIGHT PHASE: RIGHT to READY")
        self.send_arm_trajectory(
            self.right_arm_client,
            self.right_joint_names,
            self.right_current,
            self.right_ready,
            duration=3.0,
        )
        time.sleep(0.5)

        self.send_gripper(self.right_gripper_client, 0.04, left=False)
        time.sleep(0.5)

        # a) RIGHT: pick from middle (tool DOWN)
        z_mid_above_r = self.mid_z + 0.20
        z_mid_pick_r = self.mid_z + 0.03

        self.get_logger().info("RIGHT PHASE: RIGHT above middle (DOWN)")
        q_right_mid_above = self.compute_ik(
            "right_panda_arm",
            self.right_joint_names,
            self.right_current,
            self.mid_x,
            self.mid_y,
            z_mid_above_r,
            self.right_roll,
            self.right_pitch,
            self.right_yaw,
        )
        self.send_arm_trajectory(
            self.right_arm_client,
            self.right_joint_names,
            self.right_current,
            q_right_mid_above,
            duration=3.0,
        )
        time.sleep(0.5)

        self.get_logger().info("RIGHT PHASE: RIGHT down to middle (DOWN)")
        q_right_mid_pick = self.compute_ik(
            "right_panda_arm",
            self.right_joint_names,
            self.right_current,
            self.mid_x,
            self.mid_y,
            z_mid_pick_r,
            self.right_roll,
            self.right_pitch,
            self.right_yaw,
        )
        self.send_arm_trajectory(
            self.right_arm_client,
            self.right_joint_names,
            self.right_current,
            q_right_mid_pick,
            duration=2.0,
        )
        time.sleep(0.5)

        self.get_logger().info("RIGHT PHASE: close gripper at middle")
        self.send_gripper(self.right_gripper_client, 0.0, left=False)
        time.sleep(0.5)

        self.get_logger().info("RIGHT PHASE: lift from middle (DOWN)")
        self.send_arm_trajectory(
            self.right_arm_client,
            self.right_joint_names,
            self.right_current,
            q_right_mid_above,
            duration=3.0,
        )
        time.sleep(0.5)

        # b) RIGHT: swing via ARC waypoint, then to final (tool DOWN)
        z_arc_above = self.right_arc_z + 0.20
        z_final_above = self.final_z + 0.20
        z_final_place = self.final_z + 0.03

        

        # c) RIGHT: back to READY (steady)
        self.get_logger().info("RIGHT PHASE: RIGHT back to READY (steady)")
        self.send_arm_trajectory(
            self.right_arm_client,
            self.right_joint_names,
            self.right_current,
            self.right_ready,
            duration=3.0,
        )

        self.get_logger().info(
            "DONE: LEFT did start->middle then READY; "
            "RIGHT did middle->arc->final then READY."
        )


def main():
    rclpy.init()
    node = DualPickPlace()
    node.run()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
