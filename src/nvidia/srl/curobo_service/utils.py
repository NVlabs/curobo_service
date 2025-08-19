# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Utility functions."""

# Standard Library
import json
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

# Third Party
import torch
import yaml
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig

# NVIDIA
from nvidia.srl.curobo_service.data_models import (
    ForwardKinematicsRequest,
    InverseKinematicsRequest,
    JointMotionGenPlanRequest,
    MotionGenPlanRequest,
)

DEFAULT_WORLD_CONFIG = {
    "cuboid": {
        "placeholder": {
            "dims": [0.1, 0.1, 0.1],  # x, y, z
            "pose": [1000.0, 1000.0, 1000.0, 1, 0, 0, 0.0],  # x, y, z, qw, qx, qy, qz
        },
    },
}


def modify_world_config_for_mesh(world_config: str, mesh_directory: str) -> str:
    """Modify world config for mesh.

    Args:
        world_config: World configuration as a JSON string
        mesh_directory: Mesh directory

    Returns:
        Modified world configuration as a JSON string
    """
    if world_config is None or world_config == "":
        return world_config

    # Parse the JSON string to a dictionary
    config_dict = json.loads(world_config)

    if "mesh" in config_dict:
        # For each mesh in the world config, update its file_path to use the mesh_directory
        for mesh_name, mesh_data in config_dict["mesh"].items():
            if isinstance(mesh_data, dict) and "file_path" in mesh_data:
                # Get just the filename from the original path
                filename = Path(mesh_data["file_path"]).name
                # Create the new path using the mesh_directory
                mesh_data["file_path"] = str(Path(mesh_directory) / filename)

    # Convert back to JSON string
    return json.dumps(config_dict)


def create_world_config(world_config: Optional[str]) -> dict:
    """Create world config.

    Args:
        world_config: World configuration

    Returns:
        World configuration
    """
    output_dict = {}
    output_dict.update(DEFAULT_WORLD_CONFIG)
    if world_config is not None:
        # Parse world config from json
        try:
            json_config_as_dict = {}
            dict_from_json = json.loads(world_config)
            if dict_from_json:
                json_config_as_dict.update(dict_from_json)
        except Exception as e:
            raise ValueError(
                "Failed to parse world config as json, are you sure you have formatted it"
                f" correctly? config: {world_config} error: {e} dict_from_json: {dict_from_json}"
            )
        output_dict.update(json_config_as_dict)
    return output_dict


# TODO (hhadfield): Move this function to the base repo
# (see: [curobo_service](https://gitlab-master.nvidia.com/srl/py/curobo_service/-/issues/3))
def perf_timer(custom_print: Callable = print) -> Callable:
    """Decorator to measure function execution time with customized print function.

    Args:
        custom_print: Custom print function to print execution time to desired source.
    """

    def _measure_execution_time_decorator(func: Callable) -> Callable:
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            custom_print(f"{func.__name__} executed in {end - start:.6f} seconds")
            return result

        return _wrapper

    return _measure_execution_time_decorator


def convert_tensors_to_list_of_lists(tensor_object: Optional[Union[torch.Tensor, List]]) -> List:
    """Convert tensors to list of lists.

    Args:
        tensor_object: Tensor object

    Returns:
        List of lists
    """
    if tensor_object is None:
        return []
    if isinstance(tensor_object, torch.Tensor):
        return tensor_object.detach().cpu().numpy().tolist()
    if isinstance(tensor_object, list):
        return tensor_object
    return tensor_object.detach().cpu().numpy().tolist()


def convert_tensor_to_float(tensor_object: Optional[Union[torch.Tensor, float]]) -> float:
    """Convert tensor to float.

    Args:
        tensor_object: Tensor object

    Returns:
        Float
    """
    if tensor_object is None:
        return 0.0
    if issubclass(type(tensor_object), float):
        return float(tensor_object)
    else:
        if issubclass(type(tensor_object), torch.Tensor):
            return float(tensor_object.detach().cpu().numpy().item())  # type: ignore
        return float(tensor_object)


def convert_tensor_to_int(tensor_object: Optional[Union[torch.Tensor, int]]) -> int:
    """Convert tensor to int.

    Args:
        tensor_object: Tensor object

    Returns:
        Int
    """
    if tensor_object is None:
        return 0
    if issubclass(type(tensor_object), int):
        return int(tensor_object)
    else:
        if issubclass(type(tensor_object), torch.Tensor):
            return int(tensor_object.detach().cpu().numpy().item())  # type: ignore
        return int(tensor_object)


@lru_cache(maxsize=10)
def robot_config_from_urdf(
    urdf_txt: str, base_link: str, ee_link: str, tensor_args: TensorDeviceType
) -> RobotConfig:
    """Generate robot configuration from URDF.

    Args:
        urdf_txt: URDF text
        base_link: Base link
        ee_link: End-effector link
        tensor_args: Tensor device type

    Returns:
        Robot configuration
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_file = Path(tmpdirname) / "tmp.urdf"
        tmp_file.write_text(urdf_txt)
        # Generate robot configuration from  urdf path, base frame, end effector frame
        robot_cfg = RobotConfig.from_basic(tmp_file, base_link, ee_link, tensor_args)
    return robot_cfg


@lru_cache(maxsize=10)
def robot_config_from_yaml(
    yaml_txt: str,
    urdf_txt: str,
    tensor_args: TensorDeviceType,
    base_link: Optional[str],
    ee_link: Optional[str],
) -> RobotConfig:
    """Generate robot configuration from YAML.

    Args:
        yaml_txt: YAML text
        urdf_txt: URDF text
        tensor_args: Tensor device type

    Returns:
        Robot configuration
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_file = Path(tmpdirname) / "tmp.urdf"
        tmp_file.write_text(urdf_txt)
        cfg_dict = yaml.safe_load(yaml_txt)
        cfg_dict["robot_cfg"]["kinematics"]["urdf_path"] = str(tmp_file)
        if base_link is not None:
            cfg_dict["robot_cfg"]["kinematics"]["base_link"] = base_link
        if ee_link is not None:
            cfg_dict["robot_cfg"]["kinematics"]["ee_link"] = ee_link
        robot_cfg = RobotConfig.from_dict(cfg_dict, tensor_args)
    return robot_cfg


def robot_config_from_request(
    request: Union[
        ForwardKinematicsRequest,
        InverseKinematicsRequest,
        MotionGenPlanRequest,
        JointMotionGenPlanRequest,
    ],
    tensor_args: TensorDeviceType,
) -> RobotConfig:
    """Generate robot configuration from request.

    Args:
        request: Request
        tensor_args: Tensor device type

    Returns:
        Robot configuration
    """
    if request.yaml is not None:
        robot_cfg = robot_config_from_yaml(
            request.yaml,
            request.urdf,
            tensor_args,
            base_link=request.base_link,
            ee_link=request.ee_link,
        )
    else:
        if request.base_link is None:
            raise ValueError("base_link is required")
        elif request.ee_link is None:
            raise ValueError("ee_link is required")
        else:
            robot_cfg = robot_config_from_urdf(
                request.urdf, request.base_link, request.ee_link, tensor_args
            )
    return robot_cfg
