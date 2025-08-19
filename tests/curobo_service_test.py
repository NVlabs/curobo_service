# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Unit tests for the `curobo_service` package."""

# Standard Library
import json
import os
import random
import warnings
from functools import lru_cache
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Third Party
import pytest
from fastapi.testclient import TestClient

# NVIDIA
from nvidia.srl.curobo_service.data_models import (
    ForwardKinematicsRequest,
    GraspPlanRequest,
    GraspPlanResponse,
    InverseKinematicsRequest,
    InverseKinematicsResponse,
    JointMotionGenPlanRequest,
    MotionGenPlanRequest,
    MotionPlanResponse,
    RobotModelStateResponse,
)
from nvidia.srl.curobo_service.main import (
    NRM_CUROBO_PORT_DEFAULT,
    app,
    get_nrm_curobo_port,
    get_nrm_curobo_run_local,
)

FRANKA_PANDA_VALID_STARTING_POINT = [
    0.5096379518508911,
    -0.1489913910627365,
    -0.07691417634487152,
    -0.9722354412078857,
    1.2452205419540405,
    0.9569883942604065,
    0.057839129120111465,
]
UR10E_VALID_STARTING_POINT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

GOAL_END_EFFECTOR_POSITION_LIST = [
    [0.1634387969970703, 0.29817265272140503, 0.8977956175804138],
    [0.2509170472621918, 0.22182199358940125, 0.629206120967865],
]
GOAL_END_EFFECTOR_QUATERNION_LIST = [
    [
        0.44470980763435364,
        -0.47451046109199524,
        -0.7496849894523621,
        -0.12265995889902115,
    ],
    [0.08004911243915558, -0.6675625443458557, -0.7366448044776917, 0.072848379611969],
]

WORLD_OBSTACLE_CONFIG = {
    "cuboid": {
        "obstacle": {
            "dims": [0.1, 0.1, 0.1],  # x, y, z
            "pose": [0.2, 0.5, 1.2, 1, 0, 0, 0.0],  # x, y, z, qw, qx, qy, qz
        },
    },
}

WORLD_OBSTACLE_CONFIG_WITH_MESH = {
    "cuboid": {
        "obstacle": {
            "dims": [0.1, 0.1, 0.1],  # x, y, z
            "pose": [0.2, 0.5, 1.2, 1, 0, 0, 0.0],  # x, y, z, qw, qx, qy, qz
        },
    },
    "mesh": {
        "cube_mesh": {
            "file_path": "cube_mesh.obj",
            "pose": [0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0],
            "scale": [0.5, 0.5, 0.5],
        },
    },
}


class MockCollisionFreeClient:
    """Mock collision free client."""

    def __init__(self, urdf_file: PathLike, base_link: str, ee_link: str, mock_client: TestClient):
        """Initializes a new instance of the MockCollisionFreeClient class.

        Args:
            urdf_file:  The path to the URDF file.
            base_link: The name of the base link.
            ee_link: The name of the end-effector link.
            mock_client: The mock client.
        """
        self.urdf_text = Path(urdf_file).read_text()
        self.base_link = base_link
        self.ee_link = ee_link
        self.mock_client = mock_client

    def inverse_kinematics(
        self,
        ee_position: List[List],
        ee_quaternion: List[List],
    ) -> InverseKinematicsResponse:
        """Calculate the inverse kinematics for a given end-effector position and orientation.

        Args:
            ee_position: The position of the end-effector.
            ee_quaternion: The orientation of the end-effector.

        Returns:
            InverseKinematicsResponse: The response containing the calculated joint positions.
        """
        request = InverseKinematicsRequest(
            urdf=self.urdf_text,
            yaml=None,
            base_link=self.base_link,
            ee_link=self.ee_link,
            ee_position=ee_position,
            ee_quaternion=ee_quaternion,
        )
        res = self.mock_client.post(
            "inverse_kinematics",
            json=request.model_dump(),
        )
        if res.status_code == 200:
            return InverseKinematicsResponse(**res.json())
        else:
            raise Exception(res.text)

    def forward_kinematics(self, joint_position_list: List[List]) -> RobotModelStateResponse:
        """Calculates the forward kinematics for a given list of joint positions.

        Args:
            joint_position_list: The list of joint positions.

        Returns:
            RobotModelStateResponse: The response containing the calculated end-effector position,
            orientation, and the positions and orientations of the links.
        """
        request = ForwardKinematicsRequest(
            urdf=self.urdf_text,
            yaml=None,
            base_link=self.base_link,
            ee_link=self.ee_link,
            joint_position_list=joint_position_list,
        )
        res = self.mock_client.post(
            "forward_kinematics",
            json=request.model_dump(),
        )
        if res.status_code == 200:
            return RobotModelStateResponse(**res.json())
        else:
            raise Exception(res.text)

    def plan_grasp(
        self,
        start_state: List[float],
        grasp_poses: List[List[float]],
        world_config: Optional[Dict] = None,
        disable_collision_links: Optional[List[str]] = None,
        grasp_approach_offset: Optional[List[float]] = None,
        retract_offset: Optional[List[float]] = None,
        grasp_approach_path_constraint: Optional[List[float]] = None,
        retract_path_constraint: Optional[List[float]] = None,
        plan_approach_to_grasp: bool = True,
        plan_grasp_to_retract: bool = True,
        grasp_approach_constraint_in_goal_frame: bool = True,
        retract_constraint_in_goal_frame: bool = True,
    ) -> GraspPlanResponse:
        """Generates a grasp plan based on the given grasp poses.

        Args:
            start_state: The start state.
            grasp_poses: The grasp poses.
            world_config: The world configuration.
            disable_collision_links: List of links to disable collision during the final approach
            grasp_approach_offset: The offset of the grasp approach
            retract_offset: The offset of the grasp retract
            grasp_approach_path_constraint: The path constraint of the grasp approach
            retract_path_constraint: The path constraint of the grasp retract
            plan_approach_to_grasp: Whether to plan the approach to the grasp
            plan_grasp_to_retract: Whether to plan the grasp to the retract
            grasp_approach_constraint_in_goal_frame: Whether the grasp approach is in the goal frame
            retract_constraint_in_goal_frame: Whether the grasp retract is in the goal frame

        Returns:
            GraspPlanResponse: The generated grasp plan response.
        """
        assert len(grasp_poses) > 0
        assert len(grasp_poses[0]) == 7
        if grasp_approach_offset is not None:
            assert len(grasp_approach_offset) == 7
        if retract_offset is not None:
            assert len(retract_offset) == 7
        full_world_config = {} if world_config is None else world_config
        request = GraspPlanRequest(
            yaml=None,
            urdf=self.urdf_text,
            start_state=start_state,
            grasp_poses=grasp_poses,
            world_config=json.dumps(full_world_config),
            base_link=self.base_link,
            ee_link=self.ee_link,
            disable_collision_links=disable_collision_links,
            grasp_approach_offset=grasp_approach_offset,
            retract_offset=retract_offset,
            plan_approach_to_grasp=plan_approach_to_grasp,
            plan_grasp_to_retract=plan_grasp_to_retract,
            grasp_approach_constraint_in_goal_frame=grasp_approach_constraint_in_goal_frame,
            retract_constraint_in_goal_frame=retract_constraint_in_goal_frame,
            grasp_approach_path_constraint=grasp_approach_path_constraint,
            retract_path_constraint=retract_path_constraint,
        )
        res = self.mock_client.post(
            "plan_grasp",
            json=request.model_dump(),
        )
        if res.status_code == 200:
            return GraspPlanResponse(**res.json())
        else:
            raise Exception(res.text)

    def plan_single(
        self,
        goal_end_effector_position_list: List,
        goal_end_effector_quaternion_list: List,
        start_state: List[float],
        world_config: Optional[Dict] = None,
        mesh_file_list: Optional[List[str]] = None,
    ) -> MotionPlanResponse:
        """Generates a motion plan based on the given goal end-effector positions and quaternions.

        Args:
            goal_end_effector_position_list: A list of lists representing the goal
            end-effector positions.
            goal_end_effector_quaternion_list: A list of lists representing the goal
            end-effector quaternions.

        Returns:
            MotionPlanResponse: The generated motion plan response.

        """
        if mesh_file_list is not None:
            return self.plan_single_with_mesh(
                goal_end_effector_position_list,
                goal_end_effector_quaternion_list,
                start_state,
                world_config,
                mesh_file_list,
            )
        full_world_config = {} if world_config is None else world_config
        request = MotionGenPlanRequest(
            urdf=self.urdf_text,
            yaml=None,
            goal_ee_position=goal_end_effector_position_list,
            goal_ee_quaternion=goal_end_effector_quaternion_list,
            start_state=start_state,
            world_config=json.dumps(full_world_config),
            base_link=self.base_link,
            ee_link=self.ee_link,
        )
        res = self.mock_client.post(
            "motion_generation",
            json=request.model_dump(),
        )
        if res.status_code == 200:
            return MotionPlanResponse(**res.json())
        else:
            raise Exception(res.text)

    def plan_single_with_mesh(
        self,
        goal_end_effector_position_list: List,
        goal_end_effector_quaternion_list: List,
        start_state: List[float],
        world_config: Optional[Dict] = None,
        mesh_file_list: Optional[List[str]] = None,
    ) -> MotionPlanResponse:
        """Generates a motion plan based on the given goal end-effector positions and quaternions.

        Args:
            goal_end_effector_position_list: A list of lists representing the goal
            end-effector positions.
            goal_end_effector_quaternion_list: A list of lists representing the goal
            start_state: The start state of the robot.
            world_config: The world configuration.
            mesh_file_list: The list of mesh files.

        Returns:
            MotionPlanResponse: The generated motion plan response.
        """
        full_world_config = {} if world_config is None else world_config
        request = MotionGenPlanRequest(
            urdf=self.urdf_text,
            yaml=None,
            goal_ee_position=goal_end_effector_position_list,
            goal_ee_quaternion=goal_end_effector_quaternion_list,
            start_state=start_state,
            world_config=json.dumps(full_world_config),
            base_link=self.base_link,
            ee_link=self.ee_link,
        )
        files = (
            [
                (
                    "mesh_files",
                    (
                        os.path.basename(mesh_file),
                        open(mesh_file, "rb").read(),
                        "application/octet-stream",
                    ),
                )
                for mesh_file in mesh_file_list
            ]
            if mesh_file_list
            else None
        )

        # Convert the request to a JSON string and send as form data
        data = {"request": request.model_dump_json()}  # Convert to JSON string

        res = self.mock_client.post(
            "motion_generation_with_mesh",
            data=data,  # Send as form data, not JSON
            files=files,
        )
        if res.status_code == 200:
            return MotionPlanResponse(**res.json())
        else:
            raise Exception(res.text)

    def plan_single_js(
        self,
        goal_state: List[float],
        start_state: List[float],
        world_config: Optional[Dict] = None,
        mesh_file_list: Optional[List[str]] = None,
    ) -> MotionPlanResponse:
        """Generates a motion plan based on the given goal joint positions.

        Args:
            goal_stat: The goal joint positions.
            start_state: The start state of the robot.
            world_config: The world configuration.
            mesh_file_list: The list of mesh files.

        Returns:
            MotionPlanResponse: The generated motion plan response.
        """
        full_world_config = {} if world_config is None else world_config
        request = JointMotionGenPlanRequest(
            urdf=self.urdf_text,
            yaml=None,
            goal_state=goal_state,
            start_state=start_state,
            world_config=json.dumps(full_world_config),
            base_link=self.base_link,
            ee_link=self.ee_link,
        )
        if mesh_file_list is not None:
            return self.plan_single_js_with_mesh(
                goal_state,
                start_state,
                world_config,
                mesh_file_list,
            )
        res = self.mock_client.post(
            "motion_generation_js",
            json=request.model_dump(),
        )
        if res.status_code == 200:
            return MotionPlanResponse(**res.json())
        else:
            raise Exception(res.text)

    def plan_single_js_with_mesh(
        self,
        goal_state: List[float],
        start_state: List[float],
        world_config: Optional[Dict] = None,
        mesh_file_list: Optional[List[str]] = None,
    ) -> MotionPlanResponse:
        """Generates a motion plan based on the given goal joint positions.

        Args:
            goal_state: The goal joint positions.
            start_state: The start state.
            world_config: The world configuration.
            mesh_file_list: The list of mesh files.

        Returns:
            MotionPlanResponse: The generated motion plan response.
        """
        full_world_config = {} if world_config is None else world_config
        request = JointMotionGenPlanRequest(
            urdf=self.urdf_text,
            yaml=None,
            goal_state=goal_state,
            start_state=start_state,
            world_config=json.dumps(full_world_config),
            base_link=self.base_link,
            ee_link=self.ee_link,
        )
        files = (
            [
                (
                    "mesh_files",
                    (
                        os.path.basename(mesh_file),
                        open(mesh_file, "rb").read(),
                        "application/octet-stream",
                    ),
                )
                for mesh_file in mesh_file_list
            ]
            if mesh_file_list
            else None
        )

        # Convert the request to a JSON string and send as form data
        data = {"request": request.model_dump_json()}  # Convert to JSON string

        res = self.mock_client.post(
            "motion_generation_js_with_mesh",
            data=data,  # Send as form data, not JSON
            files=files,
        )
        if res.status_code == 200:
            return MotionPlanResponse(**res.json())
        else:
            raise Exception(res.text)


class MockSelfCollisionClient:
    """Mock self collision client."""

    def __init__(self, yaml_file: PathLike, urdf_file: PathLike, mock_client: TestClient):
        """Initializes a new instance of the `MockSelfCollisionClient` class.

        Args:
            yaml_file:  The path to the YAML file.
            urdf_file:  The path to the URDF file.
            mock_client: The mock client.

        Returns:
            None
        """
        self.yaml_text = Path(yaml_file).read_text()
        self.urdf_text = Path(urdf_file).read_text()
        self.mock_client = mock_client

    def inverse_kinematics(
        self,
        ee_position: List[List],
        ee_quaternion: List[List],
    ) -> InverseKinematicsResponse:
        """Calculates the inverse kinematics for a given end-effector position and orientation.

        Args:
            ee_position: The position of the end-effector.
            ee_quaternion: The orientation of the end-effector.

        Returns:
            InverseKinematicsResponse: The response containing the calculated joint positions.
        """
        request = InverseKinematicsRequest(
            yaml=self.yaml_text,
            urdf=self.urdf_text,
            ee_position=ee_position,
            ee_quaternion=ee_quaternion,
            base_link=None,
            ee_link=None,
        )
        res = self.mock_client.post(
            "inverse_kinematics",
            json=request.model_dump(),
        )
        if res.status_code == 200:
            return InverseKinematicsResponse(**res.json())
        else:
            raise Exception(res.text)

    def forward_kinematics(self, joint_position_list: List[List]) -> RobotModelStateResponse:
        """Compute forward kinematics.

        Args:
            joint_position_list: The list of joint positions.

        Returns:
            RobotModelStateResponse: The response containing the calculated end-effector position,
            orientation, and the positions and orientations of the links.
        """
        request = ForwardKinematicsRequest(
            yaml=self.yaml_text,
            urdf=self.urdf_text,
            joint_position_list=joint_position_list,
            base_link=None,
            ee_link=None,
        )
        res = self.mock_client.post(
            "forward_kinematics",
            json=request.model_dump(),
        )
        if res.status_code == 200:
            return RobotModelStateResponse(**res.json())
        else:
            raise Exception(res.text)

    def plan_grasp(
        self,
        start_state: List[float],
        grasp_poses: List[List[float]],
        world_config: Optional[Dict] = None,
        disable_collision_links: Optional[List[str]] = None,
        grasp_approach_offset: Optional[List[float]] = None,
        retract_offset: Optional[List[float]] = None,
        grasp_approach_path_constraint: Optional[List[float]] = None,
        retract_path_constraint: Optional[List[float]] = None,
        plan_approach_to_grasp: bool = True,
        plan_grasp_to_retract: bool = True,
        grasp_approach_constraint_in_goal_frame: bool = True,
        retract_constraint_in_goal_frame: bool = True,
    ) -> GraspPlanResponse:
        """Generates a grasp plan based on the given grasp poses.

        Args:
            start_state: The start state.
            grasp_poses: The grasp poses [n_grasps, 7] where 7 is [x, y, z, qw, qx, qy, qz]
            world_config: The world configuration.
            disable_collision_links: List of links to disable collision during the final approach
            grasp_approach_offset: The offset of the grasp approach [x, y, z, qw, qx, qy, qz]
            retract_offset: The offset of the grasp retract [x, y, z, qw, qx, qy, qz]
            grasp_approach_path_constraint: The path constraint of the grasp approach
            retract_path_constraint: The path constraint of the grasp retract
            plan_approach_to_grasp: Whether to plan the approach to the grasp
            plan_grasp_to_retract: Whether to plan the grasp to the retract
            grasp_approach_constraint_in_goal_frame: Whether the grasp approach is in the goal frame
            retract_constraint_in_goal_frame: Whether the grasp retract is in the goal frame

        Returns:
            GraspPlanResponse: The generated grasp plan response.
        """
        assert len(grasp_poses) > 0
        assert len(grasp_poses[0]) == 7
        if grasp_approach_offset is not None:
            assert len(grasp_approach_offset) == 7
        if retract_offset is not None:
            assert len(retract_offset) == 7
        full_world_config = {} if world_config is None else world_config
        request = GraspPlanRequest(
            yaml=self.yaml_text,
            urdf=self.urdf_text,
            start_state=start_state,
            grasp_poses=grasp_poses,
            world_config=json.dumps(full_world_config),
            base_link=None,
            ee_link=None,
            disable_collision_links=disable_collision_links,
            grasp_approach_offset=grasp_approach_offset,
            retract_offset=retract_offset,
            plan_approach_to_grasp=plan_approach_to_grasp,
            plan_grasp_to_retract=plan_grasp_to_retract,
            grasp_approach_constraint_in_goal_frame=grasp_approach_constraint_in_goal_frame,
            retract_constraint_in_goal_frame=retract_constraint_in_goal_frame,
            grasp_approach_path_constraint=grasp_approach_path_constraint,
            retract_path_constraint=retract_path_constraint,
        )
        res = self.mock_client.post(
            "plan_grasp",
            json=request.model_dump(),
        )
        if res.status_code == 200:
            return GraspPlanResponse(**res.json())
        else:
            raise Exception(res.text)

    def plan_single(
        self,
        goal_end_effector_position_list: List,
        goal_end_effector_quaternion_list: List,
        start_state: List[float],
        world_config: Optional[Dict] = None,
        mesh_file_list: Optional[List[str]] = None,
    ) -> MotionPlanResponse:
        """Generates a motion plan based on the given goal end-effector positions and quaternions.

        Args:
            goal_end_effector_position_list: A list of lists representing the goal
            end-effector positions.
            goal_end_effector_quaternion_list: A list of lists representing the goal
            end-effector quaternions.

        Returns:
            MotionPlanResponse: The generated motion plan response.
        """
        if mesh_file_list is not None:
            return self.plan_single_with_mesh(
                goal_end_effector_position_list,
                goal_end_effector_quaternion_list,
                start_state,
                world_config,
                mesh_file_list,
            )
        full_world_config = {} if world_config is None else world_config
        request = MotionGenPlanRequest(
            yaml=self.yaml_text,
            urdf=self.urdf_text,
            goal_ee_position=goal_end_effector_position_list,
            goal_ee_quaternion=goal_end_effector_quaternion_list,
            start_state=start_state,
            world_config=json.dumps(full_world_config),
            base_link=None,
            ee_link=None,
        )
        res = self.mock_client.post(
            "motion_generation",
            json=request.model_dump(),
        )
        if res.status_code == 200:
            return MotionPlanResponse(**res.json())
        else:
            raise Exception(res.text)

    def plan_single_with_mesh(
        self,
        goal_end_effector_position_list: List,
        goal_end_effector_quaternion_list: List,
        start_state: List[float],
        world_config: Optional[Dict] = None,
        mesh_file_list: Optional[List[str]] = None,
    ) -> MotionPlanResponse:
        """Generates a motion plan based on the given goal end-effector positions and quaternions.

        Args:
            goal_end_effector_position_list: A list of lists representing the goal
            end-effector positions.
            goal_end_effector_quaternion_list: A list of lists representing the goal
            end-effector quaternions.
            start_state: The start state of the robot.
            world_config: The world configuration.
            mesh_file_list: The list of mesh files.

        Returns:
            MotionPlanResponse: The generated motion plan response.
        """
        full_world_config = {} if world_config is None else world_config
        request = MotionGenPlanRequest(
            yaml=self.yaml_text,
            urdf=self.urdf_text,
            goal_ee_position=goal_end_effector_position_list,
            goal_ee_quaternion=goal_end_effector_quaternion_list,
            start_state=start_state,
            world_config=json.dumps(full_world_config),
            base_link=None,
            ee_link=None,
        )
        files = (
            [
                (
                    "mesh_files",
                    (
                        os.path.basename(mesh_file),
                        open(mesh_file, "rb").read(),
                        "application/octet-stream",
                    ),
                )
                for mesh_file in mesh_file_list
            ]
            if mesh_file_list
            else None
        )

        # Convert the request to a JSON string and send as form data
        data = {"request": request.model_dump_json()}  # Convert to JSON string

        res = self.mock_client.post(
            "motion_generation_with_mesh",
            data=data,  # Send as form data, not JSON
            files=files,
        )
        if res.status_code == 200:
            return MotionPlanResponse(**res.json())
        else:
            raise Exception(res.text)

    def plan_single_js(
        self,
        goal_state: List[float],
        start_state: List[float],
        world_config: Optional[Dict] = None,
        mesh_file_list: Optional[List[str]] = None,
    ) -> MotionPlanResponse:
        """Generates a motion plan based on the given goal joint positions.

        Args:
            goal_state: The goal joint positions.
            start_state: The start state.
            world_config: The world configuration.
            mesh_file_list: The list of mesh files.

        Returns:
            MotionPlanResponse: The generated motion plan response.
        """
        if mesh_file_list is not None:
            return self.plan_single_js_with_mesh(
                goal_state,
                start_state,
                world_config,
                mesh_file_list,
            )
        full_world_config = {} if world_config is None else world_config
        request = JointMotionGenPlanRequest(
            yaml=self.yaml_text,
            urdf=self.urdf_text,
            goal_state=goal_state,
            start_state=start_state,
            world_config=json.dumps(full_world_config),
            base_link=None,
            ee_link=None,
        )
        res = self.mock_client.post(
            "motion_generation_js",
            json=request.model_dump(),
        )
        if res.status_code == 200:
            return MotionPlanResponse(**res.json())
        else:
            raise Exception(res.text)

    def plan_single_js_with_mesh(
        self,
        goal_state: List[float],
        start_state: List[float],
        world_config: Optional[Dict] = None,
        mesh_file_list: Optional[List[str]] = None,
    ) -> MotionPlanResponse:
        """Generates a motion plan based on the given goal joint positions.

        Args:
            goal_state: The goal joint positions.
            start_state: The start state.
            world_config: The world configuration.
            mesh_file_list: The list of mesh files.

        Returns:
            MotionPlanResponse: The generated motion plan response.
        """
        full_world_config = {} if world_config is None else world_config
        request = JointMotionGenPlanRequest(
            yaml=self.yaml_text,
            urdf=self.urdf_text,
            goal_state=goal_state,
            start_state=start_state,
            world_config=json.dumps(full_world_config),
            base_link=None,
            ee_link=None,
        )
        files = (
            [
                (
                    "mesh_files",
                    (
                        os.path.basename(mesh_file),
                        open(mesh_file, "rb").read(),
                        "application/octet-stream",
                    ),
                )
                for mesh_file in mesh_file_list
            ]
            if mesh_file_list
            else None
        )

        # Convert the request to a JSON string and send as form data
        data = {"request": request.model_dump_json()}  # Convert to JSON string

        res = self.mock_client.post(
            "motion_generation_js_with_mesh",
            data=data,  # Send as form data, not JSON
            files=files,
        )
        if res.status_code == 200:
            return MotionPlanResponse(**res.json())
        else:
            raise Exception(res.text)


@lru_cache(maxsize=1)
def get_test_urdf_franka() -> Tuple[Path, str, str]:
    """Get test data parameters for Franka Panda robot.

    Returns:
        Tuple[Path, str, str]: A tuple containing the path to the URDF file, the base link,
        and the end-effector link.
    """
    this_file = Path(__file__).parent.absolute()
    urdf_file = this_file / Path("test_data/franka/franka_panda.urdf")
    base_link = "base_link"
    ee_link = "ee_link"
    return urdf_file, base_link, ee_link


@lru_cache(maxsize=1)
def get_test_yaml_franka() -> Path:
    """Returns the path to the YAML file for the Franka Panda robot.

    Returns:
        Path: The path to the YAML file.
    """
    this_file = Path(__file__).parent.absolute()
    yml_file = this_file / Path("test_data/franka/franka.yml")
    return yml_file


@lru_cache(maxsize=1)
def get_test_urdf_ur10e() -> Tuple[Path, str, str]:
    """Get test data parameters for UR10e robot.

    Returns:
        Tuple[Path, str, str]: A tuple containing the path to the URDF file, the base link,
        and the end-effector link.
    """
    this_file = Path(__file__).parent.absolute()
    urdf_file = this_file / Path("test_data/ur10e/ur10e.urdf")
    base_link = "base_link"
    ee_link = "tool0"
    return urdf_file, base_link, ee_link


@lru_cache(maxsize=1)
def get_test_yaml_ur10e() -> Path:
    """Returns the path to the YAML file for the UR10e robot.

    Returns:
        Path: The path to the YAML file.
    """
    this_file = Path(__file__).parent.absolute()
    yml_file = this_file / Path("test_data/ur10e/ur10e.yml")
    return yml_file


@lru_cache(maxsize=1)
def get_test_mesh_file() -> Path:
    """Returns the path to the mesh file for the test.

    Returns:
        Path: The path to the mesh file.
    """
    this_file = Path(__file__).parent.absolute()
    mesh_file = this_file / Path("test_data/mesh/cube_mesh.obj")
    return mesh_file


def make_test_urdf_client(robot_model: str) -> Tuple[MockCollisionFreeClient, int]:
    """Creates a test URDF client based on the given robot model.

    Args:
        robot_model: The robot model to create a client for.

    Returns:
        Tuple[MockCollisionFreeClient, int]: A tuple containing the created client and the
        number of degrees of freedom.

    Raises:
        NotImplementedError: If the given robot model is not supported.
    """
    mock_client = TestClient(app)
    if robot_model == "franka":
        urdf_file, base_link, ee_link = get_test_urdf_franka()
        client = MockCollisionFreeClient(urdf_file, base_link, ee_link, mock_client)
        dof = 7
    elif robot_model == "ur10e":
        urdf_file, base_link, ee_link = get_test_urdf_ur10e()
        client = MockCollisionFreeClient(urdf_file, base_link, ee_link, mock_client)
        dof = 6
    else:
        raise NotImplementedError
    return client, dof


def make_test_yaml_client(robot_model: str) -> Tuple[MockSelfCollisionClient, int]:
    """Creates a test client for the given robot model.

    Args:
        robot_model: The robot model for which to create a test client.

    Returns:
        Tuple[MockSelfCollisionClient, int]: A tuple containing the test client and the
        degrees of freedom (DOF) of the robot.

    Raises:
        NotImplementedError: If the given robot model is not supported.

    """
    mock_client = TestClient(app)
    if robot_model == "franka":
        yaml_file = get_test_yaml_franka()
        urdf_file = get_test_urdf_franka()[0]
        client = MockSelfCollisionClient(yaml_file, urdf_file, mock_client)
        dof = 7
    elif robot_model == "ur10e":
        yaml_file = get_test_yaml_ur10e()
        urdf_file = get_test_urdf_ur10e()[0]
        client = MockSelfCollisionClient(yaml_file, urdf_file, mock_client)
        dof = 6
    else:
        raise NotImplementedError
    return client, dof


def make_test_client(
    robot_model: str, client_type: str
) -> Tuple[Union[MockCollisionFreeClient, MockSelfCollisionClient], int]:
    """Creates a test client based on the given robot model and client type.

    Args:
        robot_model: The robot model to create a client for.
        client_type: The type of client to create.

    Returns:
        Tuple[Union[MockCollisionFreeClient, MockSelfCollisionClient], int]: A tuple containing the
        created client and the number of degrees of freedom.

    Raises:
        NotImplementedError: If the given client type is not supported.
    """
    if client_type == "urdf":
        return make_test_urdf_client(robot_model)
    elif client_type == "yaml":
        return make_test_yaml_client(robot_model)
    else:
        raise NotImplementedError


@lru_cache(maxsize=1)
def get_test_starting_point(robot_model: str) -> List[float]:
    """Returns the starting point for a given robot model.

    Args:
        robot_model: The robot model for which to get the starting point.

    Returns:
        List[float]: The starting point coordinates.

    Raises:
        NotImplementedError: If the given robot model is not supported.
    """
    if robot_model == "franka":
        return FRANKA_PANDA_VALID_STARTING_POINT
    elif robot_model == "ur10e":
        return UR10E_VALID_STARTING_POINT
    else:
        raise NotImplementedError


@pytest.mark.parametrize("robot_model", ["franka", "ur10e"])
@pytest.mark.parametrize("client_type", ["urdf", "yaml"])
def test_forward_kinematics(robot_model: str, client_type: str) -> None:
    """Test the forward kinematics functionality of the robot model.

    Args:
        robot_model: The robot model to test.
        client_type: The type of client to create.

    Returns:
        None

    Raises:
        AssertionError: If the forward kinematics response is not as expected.
    """
    fk_client, dof = make_test_client(robot_model, client_type)
    # Randomly sample several joint positions
    q = [[random.uniform(0, 1) for _ in range(dof)] for _ in range(10)]
    # Now we actually hit the server
    res = fk_client.forward_kinematics(q)
    assert res is not None
    assert isinstance(res, RobotModelStateResponse)
    assert len(res.ee_position) == 10


@pytest.mark.parametrize("robot_model", ["franka", "ur10e"])
@pytest.mark.parametrize("client_type", ["urdf", "yaml"])
def test_inverse_kinematics(robot_model: str, client_type: str) -> None:
    """Test the inverse kinematics functionality of the robot model.

    Args:
        robot_model: The robot model to test.
        client_type: The type of client to create.

    Returns:
        None

    Raises:
        AssertionError: If the inverse kinematics response is not as expected.
    """
    client, dof = make_test_client(robot_model, client_type)
    res = client.inverse_kinematics(
        GOAL_END_EFFECTOR_POSITION_LIST, GOAL_END_EFFECTOR_QUATERNION_LIST
    )
    assert res is not None
    assert isinstance(res, InverseKinematicsResponse)
    assert len(res.joint_position_list) == len(GOAL_END_EFFECTOR_POSITION_LIST)
    assert len(res.joint_position_list[0]) == dof


@pytest.mark.parametrize("robot_model", ["franka", "ur10e"])
@pytest.mark.parametrize("client_type", ["urdf", "yaml"])
@pytest.mark.parametrize(
    "world_config",
    [WORLD_OBSTACLE_CONFIG, {}, WORLD_OBSTACLE_CONFIG_WITH_MESH],
    ids=["with_obstacle", "without_obstacle", "with_obstacle_and_mesh"],
)
@pytest.mark.parametrize(
    "mesh_file_list", [None, [get_test_mesh_file()]], ids=["no_mesh", "with_mesh"]
)
def test_plan_single(
    robot_model: str, client_type: str, world_config: Dict, mesh_file_list: Optional[List[str]]
) -> None:
    """Test the `plan_single` method of the `client` object.

    Args:
        robot_model: The robot model to test.
        client_type: The type of client to create.
        world_config: The world configuration to use.

    Returns:
        None

    Raises:
        AssertionError: If the response from `plan_single` is not as expected.
    """
    if world_config == WORLD_OBSTACLE_CONFIG_WITH_MESH and mesh_file_list is None:
        pytest.skip("Skipping test because mesh file list is None")

    client, dof = make_test_client(robot_model, client_type)
    start_state = get_test_starting_point(robot_model)

    res = client.plan_single(
        GOAL_END_EFFECTOR_POSITION_LIST[0],
        GOAL_END_EFFECTOR_QUATERNION_LIST[0],
        start_state,
        world_config=world_config,
        mesh_file_list=mesh_file_list,
    )
    assert res is not None
    assert isinstance(res, MotionPlanResponse)
    assert res.success, f"res: {res}"
    assert res.position is not None
    assert len(res.position[0]) == dof


@pytest.mark.parametrize("robot_model", ["franka", "ur10e"])
@pytest.mark.parametrize("client_type", ["urdf", "yaml"])
@pytest.mark.parametrize(
    "world_config",
    [WORLD_OBSTACLE_CONFIG, {}],
    ids=["with_obstacle", "without_obstacle"],
)
@pytest.mark.parametrize(
    "grasp_approach_offset",
    [[0.0, 0.0, -0.15, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0], None],
    ids=["large_offset", "small_offset", "default_offset"],
)
@pytest.mark.parametrize(
    "retract_offset",
    [[0.0, 0.0, -0.15, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0], None],
    ids=["large_offset", "small_offset", "default_offset"],
)
def test_plan_grasp(
    robot_model: str,
    client_type: str,
    world_config: Dict,
    grasp_approach_offset: Optional[List[float]],
    retract_offset: Optional[List[float]],
) -> None:
    """Test the `plan_grasp` method of the `client` object.

    Args:
        robot_model: The robot model to test.
        client_type: The type of client to create.
        world_config: The world configuration to use.

    Returns:
        None

    Raises:
        AssertionError: If the response from `plan_grasp` is not as expected.
    """
    client, dof = make_test_client(robot_model, client_type)
    start_state = get_test_starting_point(robot_model)

    grasp_poses = [
        GOAL_END_EFFECTOR_POSITION_LIST[0] + GOAL_END_EFFECTOR_QUATERNION_LIST[0],
        GOAL_END_EFFECTOR_POSITION_LIST[0] + GOAL_END_EFFECTOR_QUATERNION_LIST[0],
    ]
    res = client.plan_grasp(
        start_state,
        grasp_poses,
        world_config=world_config,
        grasp_approach_offset=grasp_approach_offset,
        retract_offset=retract_offset,
    )
    assert res is not None
    assert isinstance(res, GraspPlanResponse)
    assert res.success, f"res: {res}"
    print(res)


@pytest.mark.parametrize("robot_model", ["franka", "ur10e"])
@pytest.mark.parametrize("client_type", ["urdf", "yaml"])
@pytest.mark.parametrize(
    "world_config",
    [WORLD_OBSTACLE_CONFIG, {}, WORLD_OBSTACLE_CONFIG_WITH_MESH],
    ids=["with_obstacle", "without_obstacle", "with_obstacle_and_mesh"],
)
@pytest.mark.parametrize(
    "mesh_file_list", [None, [get_test_mesh_file()]], ids=["no_mesh", "with_mesh"]
)
def test_plan_single_js(
    robot_model: str, client_type: str, world_config: Dict, mesh_file_list: Optional[List[str]]
) -> None:
    """Test the `plan_single_js` method of the `client` object.

    Args:
        robot_model: The robot model to test.
        client_type: The type of client to create.
        world_config: The world configuration to use.

    Returns:
        None

    Raises:
        AssertionError: If the response from `plan_single_js` is not as expected.
    """
    if world_config == WORLD_OBSTACLE_CONFIG_WITH_MESH and mesh_file_list is None:
        pytest.skip("Skipping test because mesh file list is None")

    client, dof = make_test_client(robot_model, client_type)
    start_state = get_test_starting_point(robot_model)

    # First test we just ensure it doesn't crash by planning no motion
    res = client.plan_single_js(
        start_state,
        start_state,
        world_config=world_config,
        mesh_file_list=mesh_file_list,
    )
    assert res is not None

    # Do ik to get a good starting point
    res_ik = client.inverse_kinematics(
        [GOAL_END_EFFECTOR_POSITION_LIST[0]],
        [GOAL_END_EFFECTOR_QUATERNION_LIST[0]],
    )
    assert res_ik is not None
    goal_joint_positions = res_ik.joint_position_list[0]
    # Now test that we can plan this real motion
    res = client.plan_single_js(
        goal_joint_positions,
        start_state,
        world_config=world_config,
        mesh_file_list=mesh_file_list,
    )
    assert res is not None
    assert res.success
    assert res.position is not None
    assert len(res.position[0]) == dof


def test_get_nrm_curobo_port() -> None:
    """Test the `NRM_CUROBO_PORT` function."""
    os.environ["NRM_CUROBO_PORT"] = "10002"
    nrm_curobo_port = get_nrm_curobo_port()
    assert nrm_curobo_port == 10002

    os.environ["NRM_CUROBO_PORT"] = "10001"
    nrm_curobo_port = get_nrm_curobo_port()
    assert nrm_curobo_port == 10001

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)

        os.environ["NRM_CUROBO_PORT"] = ""
        nrm_curobo_port = get_nrm_curobo_port()
        assert nrm_curobo_port == NRM_CUROBO_PORT_DEFAULT

        os.environ["NRM_CUROBO_PORT"] = "foo bar"
        nrm_curobo_port = get_nrm_curobo_port()
        assert nrm_curobo_port == NRM_CUROBO_PORT_DEFAULT


def test_get_nrm_curobo_run_local() -> None:
    """Test the `NRM_CUROBO_RUN_LOCAL` function."""
    os.environ["NRM_CUROBO_RUN_LOCAL"] = "True"
    nrm_curobo_run_local = get_nrm_curobo_run_local()
    assert nrm_curobo_run_local

    os.environ["NRM_CUROBO_RUN_LOCAL"] = "False"
    nrm_curobo_run_local = get_nrm_curobo_run_local()
    assert not nrm_curobo_run_local

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)

        os.environ["NRM_CUROBO_RUN_LOCAL"] = ""
        nrm_curobo_run_local = get_nrm_curobo_run_local()
        assert not nrm_curobo_run_local

        os.environ["NRM_CUROBO_RUN_LOCAL"] = "foo bar"
        nrm_curobo_run_local = get_nrm_curobo_run_local()
        assert not nrm_curobo_run_local


if __name__ == "__main__":
    pytest.main()
