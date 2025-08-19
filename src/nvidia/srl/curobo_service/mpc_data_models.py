# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""NVIDIA Curobo Service Model Predictive Control data models."""

# Standard Library
from typing import List, Optional

# Third Party
from pydantic import BaseModel


class JointStateModel(BaseModel):
    """Joint State Model.

    Args:
        position: The joint positions.
        velocity: The joint velocities.
        acceleration: The joint accelerations.
        jerk: The joint jerks.
        joint_names: The joint names.
    """

    position: List[float]
    velocity: List[float]
    acceleration: List[float]
    jerk: List[float]
    joint_names: List[str]


class ArmReacherMetricsModel(BaseModel):
    """Arm Reacher Metrics Model.

    Args:
        cost: The cost.
        constraint: The constraint.
        feasible: The feasible.
        position_error: The position error.
        rotation_error: The rotation error.
        pose_error: The pose error.
        goalset_index: The goalset index.
        null_space_error: The null space error.
    """

    cost: List[List[float]]
    constraint: List[List[float]]
    feasible: List[List[bool]]
    position_error: List[List[float]]
    rotation_error: List[List[float]]
    pose_error: List[List[float]]
    goalset_index: List[List[int]]
    null_space_error: List[List[float]]


class StepMpcResponse(BaseModel):
    """Step Mpc Response.

    Args:
        solve_time: The solve time.
        action: The action.
        metrics: The metrics.
    """

    solve_time: float
    action: JointStateModel
    metrics: ArmReacherMetricsModel


class NewMpcRequest(BaseModel):
    """New Mpc Request.

    Args:
        yaml: The YAML text.
        urdf: The URDF text.
        world_config: The world configuration.
        step_dt: The timestep in seconds.
    """

    yaml: str
    urdf: str
    world_config: str
    step_dt: float


class NewMpcResponse(BaseModel):
    """New Mpc Response.

    Args:
        success: The success.
        mpc_id: The mpc id.
        joint_names: The joint names.
    """

    success: bool
    mpc_id: int
    joint_names: List[str]


class SetGoalPoseRequest(BaseModel):
    """Set Goal Pose Request.

    Args:
        ee_position: The end-effector position [x, y, z].
        ee_quaternion: The end-effector orientation [w, x, y, z].
        extra_link_names: The extra link names.
        extra_link_positions: The extra link positions.
        extra_link_quaternions: The extra link quaternions.
    """

    ee_position: List[float]
    ee_quaternion: List[float]
    extra_link_names: Optional[List[str]]
    extra_link_positions: Optional[List[List[float]]]
    extra_link_quaternions: Optional[List[List[float]]]


class InitializeGoalRequest(BaseModel):
    """Initialize Goal Request.

    Args:
        start_state_joint_position: The start state joint position [j1, ..., jN].
        goal_position: The goal position [x, y, z].
        goal_quaternion: The goal quaternion [w, x, y, z].
        extra_link_names: The extra link names.
        extra_link_positions: The extra link positions.
        extra_link_quaternions: The extra link quaternions.
    """

    start_state_joint_position: List[float]
    goal_position: List[float]
    goal_quaternion: List[float]
    extra_link_names: Optional[List[str]]
    extra_link_positions: Optional[List[List[float]]]
    extra_link_quaternions: Optional[List[List[float]]]
