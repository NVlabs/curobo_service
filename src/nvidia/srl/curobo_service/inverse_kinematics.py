# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Inverse Kinematics functions."""

# Standard Library
from typing import List

# Third Party
import torch

# cuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

# NVIDIA
# cuRobo Service
from nvidia.srl.curobo_service.data_models import (
    InverseKinematicsRequest,
    InverseKinematicsResponse,
)
from nvidia.srl.curobo_service.utils import (
    convert_tensors_to_list_of_lists,
    robot_config_from_request,
)


def compute_inverse_kinematics(
    robot_cfg: RobotConfig,
    tensor_args: TensorDeviceType,
    ee_position: List[List[float]],
    ee_quaternion: List[List[float]],
) -> InverseKinematicsResponse:
    """Compute inverse kinematics.

    Args:
        robot_cfg: Robot configuration.
        tensor_args: Tensor device type.
        ee_position: End-effector position.
        ee_quaternion: End-effector quaternion.

    Returns:
        Inverse Kinematics Response.
    """
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        None,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=20,
        self_collision_check=False,
        self_collision_opt=False,
        tensor_args=tensor_args,
        use_cuda_graph=True,
    )
    ik_solver = IKSolver(ik_config)
    ee_position_tensor = torch.tensor(ee_position, **(tensor_args.as_torch_dict()))
    ee_quaternion_tensor = torch.tensor(ee_quaternion, **(tensor_args.as_torch_dict()))
    goal = Pose(ee_position_tensor, ee_quaternion_tensor)
    result = ik_solver.solve_batch(goal)
    joint_position_list = [tmp[0] for tmp in convert_tensors_to_list_of_lists(result.solution)]
    success = [s[0] for s in result.success.cpu().numpy().tolist()]
    return InverseKinematicsResponse(joint_position_list=joint_position_list, success=success)


def compute_inverse_kinematics_request(
    request: InverseKinematicsRequest,
) -> InverseKinematicsResponse:
    """Compute inverse kinematics with URDF.

    Args:
        request: Inverse Kinematics Request made with URDF.

    Returns:
        Inverse Kinematics Response.
    """
    tensor_args = TensorDeviceType()
    robot_config = robot_config_from_request(request, tensor_args)
    return compute_inverse_kinematics(
        robot_config,
        tensor_args,
        request.ee_position,
        request.ee_quaternion,
    )
