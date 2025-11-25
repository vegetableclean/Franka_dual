#!/usr/bin/env python3
import time
import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK, GetPositionIK_Request, ApplyPlanningScene
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory

from moveit_msgs.msg import PlanningScene, CollisionObject, AttachedCollisionObject
from shape_msgs.msg import SolidPrimitive


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

        # ===== MoveIt planning scene client =====
        self.ps_client = self.create_client(
            ApplyPlanningScene, "/apply_planning_scene"
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

        # Final place pose for RIGHT (right side, easier IK)
        self.final_x = 0.45
        self.final_y = 0.25
        self.final_z = 0.00    # table height

        # Object box settings
        self.object_id = "dual_panda_box"
        self.box_size = (0.05, 0.05, 0.05)  # 5cm cube

        # EE link names (from your xacro: left_panda_hand / right_panda_hand)
        self.left_ee_link = "left_panda_hand"
        self.right_ee_link = "right_panda_hand"

    # ======================================================
    #   Helpers
    # ======================================================
    def wait_for_servers(self):
        # IK
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /compute_ik...")

        # Arms
        self.get_logger().info("Waiting for left arm controller...")
        self.left_arm_client.wait_for_server()
        self.get_logger().info("Waiting for right arm controller...")
        self.right_arm_client.wait_for_server()

        # Grippers
        self.get_logger().info("Waiting for left gripper controller...")
        self.left_gripper_client.wait_for_server()
        self.get_logger().info("Waiting for right gripper controller...")
        self.right_gripper_client.wait_for_server()

        # Planning scene
        self.get_logger().info("Waiting for /apply_planning_scene...")
        while not self.ps_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("  ... still waiting for /apply_planning_scene")

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
        pose.header.stamp = self.get_clock().now().to_msg()
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
        req.ik_request.avoid_collisions = False  # debugging: ignore collisions

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
    #   Planning scene helpers (box object)
    # ======================================================
    def _make_box_collision_object(self, frame_id, x, y, z):
        co = CollisionObject()
        co.id = self.object_id
        co.header.frame_id = frame_id

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = list(self.box_size)  # [x, y, z]

        # pose: box center at (x, y, z + box_height/2) if frame is world
        pose = PoseStamped().pose
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z + self.box_size[2] / 2.0)
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0

        co.primitives.append(box)
        co.primitive_poses.append(pose)
        co.operation = CollisionObject.ADD
        return co

    def _apply_planning_scene(self, planning_scene: PlanningScene):
        req = ApplyPlanningScene.Request()
        req.scene = planning_scene
        future = self.ps_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()
        if res is None:
            self.get_logger().error("ApplyPlanningScene returned None")
        else:
            self.get_logger().info("Planning scene updated.")

    def add_box_in_world(self, x, y, z):
        """Add the box as a world collision object at (x, y, z) on the table."""
        co = self._make_box_collision_object("world", x, y, z)
        ps = PlanningScene()
        ps.is_diff = True
        ps.world.collision_objects.append(co)
        self.get_logger().info(
            f"Adding world box at x={x:.3f}, y={y:.3f}, z={z:.3f}"
        )
        self._apply_planning_scene(ps)

    def attach_box_to_link(self, link_name):
        """
        Remove the box from the world and attach it to link_name
        so it moves with the gripper in RViz.
        """
        # Remove from world
        co_remove = CollisionObject()
        co_remove.id = self.object_id
        co_remove.header.frame_id = "world"
        co_remove.operation = CollisionObject.REMOVE

        # Attach to robot
        aco = AttachedCollisionObject()
        aco.link_name = link_name
        aco.touch_links = [link_name]  # simple: only allow contact with EE link

        # object relative to the link frame
        co = CollisionObject()
        co.id = self.object_id
        co.header.frame_id = link_name

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = list(self.box_size)

        pose = PoseStamped().pose
        # Slight offset in +z of the link (adjust if needed to align with fingers)
        pose.position.x = 0.0
        pose.position.y = 0.0
        pose.position.z = self.box_size[2] / 2.0
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = 0.0
        pose.orientation.w = 1.0

        co.primitives.append(box)
        co.primitive_poses.append(pose)
        co.operation = CollisionObject.ADD

        aco.object = co

        ps = PlanningScene()
        ps.is_diff = True
        ps.world.collision_objects.append(co_remove)
        ps.robot_state.attached_collision_objects.append(aco)

        self.get_logger().info(f"Attaching box to link {link_name}")
        self._apply_planning_scene(ps)

    def detach_box_to_world(self, x, y, z, from_link_name):
        """
        Detach from the given link and spawn back as world object at (x,y,z).
        """
        # Remove attached
        aco_remove = AttachedCollisionObject()
        aco_remove.link_name = from_link_name
        aco_remove.object = CollisionObject()
        aco_remove.object.id = self.object_id
        aco_remove.object.operation = CollisionObject.REMOVE

        # Add back to world at new pose
        co_add = self._make_box_collision_object("world", x, y, z)

        ps = PlanningScene()
        ps.is_diff = True
        ps.robot_state.attached_collision_objects.append(aco_remove)
        ps.world.collision_objects.append(co_add)

        self.get_logger().info(
            f"Detaching box from {from_link_name} to world at "
            f"x={x:.3f}, y={y:.3f}, z={z:.3f}"
        )
        self._apply_planning_scene(ps)

    # ======================================================
    #   Main sequence
    # ======================================================
    def run(self):
        self.wait_for_servers()

        # 1) Put the box in the world at START position on the table
        self.add_box_in_world(self.start_x, self.start_y, self.start_z)
        time.sleep(0.5)

        # ---------- PHASE 1: LEFT ONLY ----------

        # Ensure LEFT and RIGHT are in READY
        self.get_logger().info("LEFT PHASE: move LEFT to READY...")
        self.send_arm_trajectory(
            self.left_arm_client,
            self.left_joint_names,
            self.left_current,
            self.left_ready,
            duration=3.0,
        )
        self.get_logger().info("RIGHT PHASE: RIGHT to READY (steady)")
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

        # Attach box to left gripper so it follows LEFT
        self.attach_box_to_link(self.left_ee_link)
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

        # Detach box to world at MIDDLE pose
        self.detach_box_to_world(self.mid_x, self.mid_y, self.mid_z, self.left_ee_link)
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

        # ---------- PHASE 2: RIGHT ONLY ----------

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

        # Attach box to RIGHT gripper
        self.attach_box_to_link(self.right_ee_link)
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

        # b) RIGHT: move to final and drop (DOWN)
        z_final_above = self.final_z + 0.20
        z_final_place = self.final_z + 0.03

        self.get_logger().info("RIGHT PHASE: RIGHT above final (DOWN)")
        q_right_final_above = self.compute_ik(
            "right_panda_arm",
            self.right_joint_names,
            self.right_current,
            self.final_x,
            self.final_y,
            z_final_above,
            self.right_roll,
            self.right_pitch,
            self.right_yaw,
        )
        self.send_arm_trajectory(
            self.right_arm_client,
            self.right_joint_names,
            self.right_current,
            q_right_final_above,
            duration=3.0,
        )
        time.sleep(0.5)

        self.get_logger().info("RIGHT PHASE: RIGHT down to final (DOWN)")
        q_right_final_place = self.compute_ik(
            "right_panda_arm",
            self.right_joint_names,
            self.right_current,
            self.final_x,
            self.final_y,
            z_final_place,
            self.right_roll,
            self.right_pitch,
            self.right_yaw,
        )
        self.send_arm_trajectory(
            self.right_arm_client,
            self.right_joint_names,
            self.right_current,
            q_right_final_place,
            duration=2.0,
        )
        time.sleep(0.5)

        self.get_logger().info("RIGHT PHASE: open gripper at final")
        self.send_gripper(self.right_gripper_client, 0.04, left=False)
        time.sleep(0.5)

        # Detach box to world at FINAL pose
        self.detach_box_to_world(
            self.final_x, self.final_y, self.final_z, self.right_ee_link
        )
        time.sleep(0.5)

        self.get_logger().info("RIGHT PHASE: lift from final (DOWN)")
        self.send_arm_trajectory(
            self.right_arm_client,
            self.right_joint_names,
            self.right_current,
            q_right_final_above,
            duration=3.0,
        )
        time.sleep(0.5)

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
            "RIGHT did middle->final then READY, with box attached/detached in MoveIt."
        )


def main():
    rclpy.init()
    node = DualPickPlace()
    node.run()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
