# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Forward kinematics functions."""

# Standard Library
from typing import List

# Third Party
import torch

# cuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelState
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig

# NVIDIA
# cuRobo Service
from nvidia.srl.curobo_service.data_models import ForwardKinematicsRequest, RobotModelStateResponse
from nvidia.srl.curobo_service.utils import (
    convert_tensors_to_list_of_lists,
    robot_config_from_request,
)


def convert_to_robot_model_state_response(
    robot_state: CudaRobotModelState,
) -> RobotModelStateResponse:
    """Convert CudaRobotModelState to RobotModelStateResponse.

    Args:
        robot_state: CudaRobotModelState.

    Returns:
        RobotModelStateResponse.
    """
    output = RobotModelStateResponse(
        **{  # type: ignore
            "ee_position": convert_tensors_to_list_of_lists(robot_state.ee_position),
            "ee_quaternion": convert_tensors_to_list_of_lists(robot_state.ee_quaternion),
            "links_position": convert_tensors_to_list_of_lists(robot_state.links_position),
            "links_quaternion": convert_tensors_to_list_of_lists(robot_state.links_quaternion),
            "link_spheres_tensor": convert_tensors_to_list_of_lists(
                robot_state.link_spheres_tensor
            ),
            "link_names": robot_state.link_names,
        }
    )
    return output


def compute_forward_kinematics(
    robot_cfg: RobotConfig,
    joint_position_list: List[List[float]],
    tensor_args: TensorDeviceType,
) -> RobotModelStateResponse:
    """Compute forward kinematics.

    Args:
        robot_cfg: Robot configuration.
        joint_position_list: Joint position list.
        tensor_args: Tensor device type.

    Returns:
        Robot Model State Response.
    """
    kin_model = CudaRobotModel(robot_cfg.kinematics)
    joint_position_tensor = torch.tensor(joint_position_list, **(tensor_args.as_torch_dict()))
    # compute forward kinematics:
    tensor_out = kin_model.get_state(joint_position_tensor)
    return convert_to_robot_model_state_response(tensor_out)


def compute_forward_kinematics_request(
    request: ForwardKinematicsRequest,
) -> RobotModelStateResponse:
    """Compute forward kinematics with URDF.

    Args:
        request: Forward Kinematics Request made with URDF.

    Returns:
        Robot Model State Response.
    """
    # convenience function to store tensor type and device
    tensor_args = TensorDeviceType()
    robot_config = robot_config_from_request(request, tensor_args)
    return compute_forward_kinematics(robot_config, request.joint_position_list, tensor_args)
