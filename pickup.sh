#!/bin/bash
set -e

# Simple left-arm pick sequence using FollowJointTrajectory

# --- Step 0: Move arm to READY pose ---
echo "=== Step 0: Move arm to READY pose ==="
ros2 action send_goal /left_arm_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory "
trajectory:
  joint_names:
    - left_panda_joint1
    - left_panda_joint2
    - left_panda_joint3
    - left_panda_joint4
    - left_panda_joint5
    - left_panda_joint6
    - left_panda_joint7
  points:
    - positions: [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
      time_from_start:
        sec: 2
        nanosec: 0
"
sleep 3

# --- Step 1: Move arm above object (pre-grasp pose) ---
# These joint values are an example that bring the hand forward and somewhat low.
echo "=== Step 1: Move arm above object ==="
ros2 action send_goal /left_arm_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory "
trajectory:
  joint_names:
    - left_panda_joint1
    - left_panda_joint2
    - left_panda_joint3
    - left_panda_joint4
    - left_panda_joint5
    - left_panda_joint6
    - left_panda_joint7
  points:
    - positions: [0.2, -1.0, 0.2, -2.2, 0.1, 1.8, 0.5]
      time_from_start:
        sec: 3
        nanosec: 0
"
sleep 4

# --- Step 2: Open gripper ---
echo "=== Step 2: Open gripper ==="
ros2 action send_goal /left_gripper_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory "
trajectory:
  joint_names:
    - left_panda_finger_joint1
  points:
    - positions: [0.04]   # Open
      time_from_start:
        sec: 1
        nanosec: 0
"
sleep 2

# --- Step 3: Move arm down towards object ---
# Slight change in joint2 and joint4 to lower the end effector.
echo "=== Step 3: Move arm down to object ==="
ros2 action send_goal /left_arm_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory "
trajectory:
  joint_names:
    - left_panda_joint1
    - left_panda_joint2
    - left_panda_joint3
    - left_panda_joint4
    - left_panda_joint5
    - left_panda_joint6
    - left_panda_joint7
  points:
    - positions: [0.2, -1.3, 0.2, -2.6, 0.1, 1.8, 0.5]
      time_from_start:
        sec: 2
        nanosec: 0
"
sleep 3

# --- Step 4: Close gripper to grasp ---
echo "=== Step 4: Close gripper ==="
ros2 action send_goal /left_gripper_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory "
trajectory:
  joint_names:
    - left_panda_finger_joint1
  points:
    - positions: [0.0]   # Close
      time_from_start:
        sec: 1
        nanosec: 0
"
sleep 2

# --- Step 5: Lift arm with object (back to pre-grasp pose) ---
echo "=== Step 5: Lift arm with object ==="
ros2 action send_goal /left_arm_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory "
trajectory:
  joint_names:
    - left_panda_joint1
    - left_panda_joint2
    - left_panda_joint3
    - left_panda_joint4
    - left_panda_joint5
    - left_panda_joint6
    - left_panda_joint7
  points:
    - positions: [0.2, -1.0, 0.2, -2.2, 0.1, 1.8, 0.5]
      time_from_start:
        sec: 3
        nanosec: 0
"
sleep 4

echo "=== DONE ==="

