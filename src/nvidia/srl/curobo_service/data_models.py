# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Data models for motion planning."""

# Standard Library
from typing import List, Optional

# Third Party
from pydantic import BaseModel


class MpcStatus(BaseModel):
    """Response to information about the status of an MPC session."""

    status: str


class HealthCheck(BaseModel):
    """Response model to validate and return when performing a health check."""

    status: str


class ForwardKinematicsRequest(BaseModel):
    """Forward Kinematics Request."""

    urdf: str
    yaml: Optional[str]
    base_link: Optional[str]
    ee_link: Optional[str]
    joint_position_list: List[List[float]]


class InverseKinematicsRequest(BaseModel):
    """Inverse Kinematics Request."""

    urdf: str
    yaml: Optional[str]
    base_link: Optional[str]
    ee_link: Optional[str]
    ee_position: List[List[float]]
    ee_quaternion: List[List[float]]


class MotionGenPlanRequest(BaseModel):
    """Motion Generation Plan Request."""

    urdf: str
    yaml: Optional[str]
    base_link: Optional[str]
    ee_link: Optional[str]
    goal_ee_position: List[float]
    goal_ee_quaternion: List[float]
    start_state: List[float]
    world_config: Optional[str]


class JointMotionGenPlanRequest(BaseModel):
    """Joint Motion Generation Plan Request."""

    urdf: str
    yaml: Optional[str]
    base_link: Optional[str]
    ee_link: Optional[str]
    goal_state: List[float]
    start_state: List[float]
    world_config: Optional[str]


class MotionPlanResponse(BaseModel):
    """Motion Plan Response."""

    success: bool
    dt: float
    status: Optional[str]
    position: Optional[List[List[float]]]
    velocity: Optional[List[List[float]]]
    acceleration: Optional[List[List[float]]]
    jerk: Optional[List[List[float]]]


class RobotModelStateResponse(BaseModel):
    """Robot Model State Response."""

    ee_position: List[List[float]]
    ee_quaternion: List[List[float]]
    links_position: Optional[List[List[List[float]]]]
    links_quaternion: Optional[List[List[List[float]]]]
    link_spheres_tensor: Optional[List[List[List[float]]]]
    link_names: Optional[List[str]]


class InverseKinematicsResponse(BaseModel):
    """Inverse Kinematics Response."""

    joint_position_list: List[List[float]]
    success: List[int]


class JointStateTrajectory(BaseModel):
    """Joint State."""

    position: List[List[float]]
    velocity: List[List[float]]
    acceleration: List[List[float]]


class GraspPlanResponse(BaseModel):
    """Grasp Plan Response."""

    success: Optional[List[bool]] = None
    grasp_trajectory: Optional[JointStateTrajectory] = None
    grasp_trajectory_dt: Optional[float] = None
    grasp_interpolated_trajectory: Optional[JointStateTrajectory] = None
    grasp_interpolation_dt: Optional[float] = None
    retract_trajectory: Optional[JointStateTrajectory] = None
    retract_trajectory_dt: Optional[float] = None
    retract_interpolated_trajectory: Optional[JointStateTrajectory] = None
    retract_interpolation_dt: Optional[float] = None
    status: Optional[str] = None
    planning_time: float = 0.0
    goalset_index: Optional[int] = None


class GraspPlanRequest(BaseModel):
    """Grasp Plan Request.

    Args:
        urdf: The URDF text.
        yaml: The YAML text.
        base_link: The base link.
        ee_link: The end-effector link.
        world_config: The world configuration.
        start_state: The start state.
        grasp_poses: The grasp poses of shape [n_grasps, 7] where 7 is [x, y, z, qw, qx, qy, qz]
        disable_collision_links: The list of links to disable collision during the final approach
        grasp_approach_offset: The offset of the grasp approach [x, y, z, qw, qx, qy, qz]
        retract_offset: The offset of the grasp retract [x, y, z, qw, qx, qy, qz]
    """

    urdf: str
    yaml: Optional[str]
    base_link: Optional[str]
    ee_link: Optional[str]
    world_config: Optional[str]
    start_state: List[float]
    grasp_poses: List[List[float]]
    disable_collision_links: Optional[List[str]]
    grasp_approach_offset: Optional[List[float]]
    retract_offset: Optional[List[float]]
    grasp_approach_path_constraint: Optional[List[float]]
    retract_path_constraint: Optional[List[float]]
    plan_approach_to_grasp: bool
    plan_grasp_to_retract: bool
    grasp_approach_constraint_in_goal_frame: bool
    retract_constraint_in_goal_frame: bool
