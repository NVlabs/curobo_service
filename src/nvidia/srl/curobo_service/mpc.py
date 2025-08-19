# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""NVIDIA Curobo Service Model Predictive Control API."""


# Standard Library
from typing import Dict, List, Optional

# Third Party
import torch
import yaml
from curobo.geom.types import WorldConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig

# NVIDIA
from nvidia.srl.curobo_service.mpc_data_models import (
    ArmReacherMetricsModel,
    JointStateModel,
    StepMpcResponse,
)
from nvidia.srl.curobo_service.utils import create_world_config, robot_config_from_yaml


class MpcManager:
    """NVIDIA Curobo Service Model Predictive Control Manager."""

    def __init__(self, config: MpcSolverConfig, tensor_args: TensorDeviceType):
        """Initialize the MPC manager.

        Args:
            config: The MPC solver configuration.
            tensor_args: The tensor args.
        """
        self.mpc = MpcSolver(config)
        self.tensor_args = tensor_args

    @staticmethod
    def from_yaml(
        yaml_txt: str, urdf_str: str, world_config_str: str, step_dt: float
    ) -> "MpcManager":
        """Create the MPC manager from YAML.

        Args:
            yaml_txt: The YAML text.
            urdf_str: The URDF text.
            world_config: The world configuration.
        step_dt: The timestep in seconds.

        Returns:
            The MPC manager.
        """
        world_config = yaml.safe_load(world_config_str)
        tensor_args = TensorDeviceType()
        robot_cfg = robot_config_from_yaml(yaml_txt, urdf_str, tensor_args, None, None)
        config = MpcSolverConfig.load_from_robot_config(
            robot_cfg=robot_cfg,
            world_model=world_config,
            step_dt=step_dt,
            use_cuda_graph=True,
        )
        return MpcManager(config, tensor_args)

    def reset(self) -> None:
        """Reset the MPC solver."""
        self.mpc.reset()

    def update_current_joint_state_list_positions(self, joint_state_list: List[float]) -> None:
        """Update the current joint state from a list of positions.

        Args:
            joint_state_list: The new joint state list [j1, ..., jN].
        """
        js = JointState.from_position(
            torch.tensor(joint_state_list, **(self.tensor_args.as_torch_dict())),
            joint_names=self.mpc.joint_names,
        )
        self.update_current_joint_state(js)

    def update_current_joint_state_request(self, request: JointStateModel) -> None:
        """Update the current joint state from a request.

        Args:
            request: The new joint state request.
        """
        js_obj = JointState(
            torch.tensor(request.position, **(self.mpc.tensor_args.as_torch_dict())),
            velocity=torch.tensor(request.velocity, **(self.mpc.tensor_args.as_torch_dict())),
            acceleration=torch.tensor(
                request.acceleration, **(self.mpc.tensor_args.as_torch_dict())
            ),
            joint_names=self.mpc.joint_names,
        )
        self.update_current_joint_state(js_obj)

    def update_current_joint_state(self, joint_state: JointState) -> None:
        """Update the current joint state.

        Args:
            joint_state: The new joint state.
        """
        self.current_joint_state = joint_state

    def setup_goal_from_position_and_quaternion(
        self,
        start_state_list: List[float],
        goal_position: List[float],
        goal_quaternion: List[float],
        link_poses_names: Optional[List[str]] = None,
        link_poses_goal_positions: Optional[List[List[float]]] = None,
        link_poses_goal_quaternions: Optional[List[List[float]]] = None,
    ) -> None:
        """Setup the goal from a position and a quaternion.

        Args:
            start_state_list: The start state joint position [j1, ..., jN].
            goal_position: The goal position [x, y, z].
            goal_quaternion: The goal quaternion [w, x, y, z].
            link_poses_names: The link poses names.
            link_poses_goal_positions: The link poses goal positions.
            link_poses_goal_quaternions: The link poses goal quaternions.
        """
        start_state = JointState.from_position(
            torch.tensor(start_state_list, **(self.tensor_args.as_torch_dict())),
            joint_names=self.mpc.joint_names,
        )
        goal_pose = Pose(
            position=torch.tensor(goal_position, **(self.tensor_args.as_torch_dict())),
            quaternion=torch.tensor(goal_quaternion, **(self.tensor_args.as_torch_dict())),
        )
        # Construct link poses and add them to the goal pose
        link_poses: Optional[Dict[str, Pose]] = None
        if (
            link_poses_names is not None
            and link_poses_goal_positions is not None
            and link_poses_goal_quaternions is not None
        ):
            link_poses = {}
            for lp_name, lp_goal_position, lp_goal_quaternion in zip(
                link_poses_names, link_poses_goal_positions, link_poses_goal_quaternions
            ):
                link_poses[lp_name] = Pose(
                    position=torch.tensor(lp_goal_position, **(self.tensor_args.as_torch_dict())),
                    quaternion=torch.tensor(
                        lp_goal_quaternion, **(self.tensor_args.as_torch_dict())
                    ),
                )
        self.setup_goal(start_state=start_state, goal_pose=goal_pose, link_poses=link_poses)

    def setup_goal_from_position_and_pose(
        self, start_state_list: List[float], goal_pose_list: List[float]
    ) -> None:
        """Setup the goal from a position and a pose.

        Args:
            start_state_list: The start state joint position [j1, ..., jN].
            goal_pose_list: The goal pose [x, y, z, w, x, y, z].
        """
        start_state = JointState.from_position(
            torch.tensor(start_state_list, **(self.tensor_args.as_torch_dict())),
            joint_names=self.mpc.joint_names,
        )
        goal_pose = Pose.from_list(goal_pose_list)
        self.setup_goal(start_state=start_state, goal_pose=goal_pose)

    def setup_goal(
        self, start_state: JointState, goal_pose: Pose, link_poses: Optional[Dict[str, Pose]] = None
    ) -> None:
        """Setup the goal pose.

        Args:
            start_state: The start state.
            goal_pose: The goal pose.
            link_poses: The link poses.
        """
        goal = Goal(
            current_state=start_state,
            goal_pose=goal_pose,
            links_goal_pose=link_poses,
        )
        self.setup_goal_buffer(goal)
        self.update_current_joint_state(start_state)

    def setup_goal_buffer(self, goal: Goal) -> None:
        """Setup the goal buffer.

        Args:
            goal: The goal.
        """
        self._goal_buffer = self.mpc.setup_solve_single(goal, 1)

    def update_goal_pose_from_position_and_quaternion(
        self,
        goal_position: List[float],
        goal_quaternion: List[float],
        link_poses_names: Optional[List[str]] = None,
        link_poses_goal_positions: Optional[List[List[float]]] = None,
        link_poses_goal_quaternions: Optional[List[List[float]]] = None,
    ) -> None:
        """Update the goal pose from a position and a quaternion [w, x, y, z].

        Args:
            goal_position: The goal position [x, y, z].
            goal_quaternion: The goal quaternion [w, x, y, z].
            link_poses_names: The link poses names.
            link_poses_goal_positions: The link poses goal positions.
            link_poses_goal_quaternions: The link poses goal quaternions.
        """
        position_tensor = torch.tensor(goal_position, **(self.tensor_args.as_torch_dict()))
        quaternion_tensor = torch.tensor(goal_quaternion, **(self.tensor_args.as_torch_dict()))
        goal_pose = Pose(position=position_tensor, quaternion=quaternion_tensor)
        # Construct link poses and add them to the goal pose
        link_poses: Optional[Dict[str, Pose]] = None
        if (
            link_poses_names is not None
            and link_poses_goal_positions is not None
            and link_poses_goal_quaternions is not None
        ):
            link_poses = {}
            for lp_name, lp_position, lp_quaternion in zip(
                link_poses_names, link_poses_goal_positions, link_poses_goal_quaternions
            ):
                link_poses[lp_name] = Pose(
                    position=torch.tensor(lp_position, **(self.tensor_args.as_torch_dict())),
                    quaternion=torch.tensor(lp_quaternion, **(self.tensor_args.as_torch_dict())),
                )
        goal = Goal(
            current_state=self.current_joint_state,
            goal_pose=goal_pose,
            links_goal_pose=link_poses,
        )
        self.update_goal(goal)

    def update_goal_pose_from_list(self, goal_pose_list: List[float]) -> None:
        """Update the goal pose from a list.

        Args:
            goal_pose_list: The new goal pose list [x, y, z, w, x, y, z].
        """
        goal_pose = Pose.from_list(goal_pose_list)
        self.update_goal_pose(goal_pose)

    def update_goal(self, goal: Goal) -> None:
        """Update the goal, including goal pose and link poses."""
        self._goal_buffer.goal_pose.copy_(goal.goal_pose)
        if goal.links_goal_pose is not None:
            if self._goal_buffer.links_goal_pose is None:
                self._goal_buffer.links_goal_pose = goal.links_goal_pose
            else:
                for k in goal.links_goal_pose.keys():
                    self._goal_buffer.links_goal_pose[k] = self._goal_buffer._copy_buffer(
                        self._goal_buffer.links_goal_pose[k], goal.links_goal_pose[k]
                    )
        self._goal_buffer._update_batch_size()
        self.mpc.update_goal(self._goal_buffer)
        self.mpc.enable_cspace_cost(enable=False)
        self.mpc.enable_pose_cost(enable=True)

    def update_goal_pose(self, new_pose: Pose) -> None:
        """Update the goal pose.

        Args:
            new_pose: The new goal pose.
        """
        self._goal_buffer.goal_pose.copy_(new_pose)
        self.mpc.update_goal(self._goal_buffer)
        self.mpc.enable_cspace_cost(enable=False)
        self.mpc.enable_pose_cost(enable=True)

    def update_world(self, world_str: str) -> None:
        """Update the world.

        Args:
            world_str: The new world configuration as a string.
        """
        world_config = WorldConfig.from_dict(create_world_config(world_str))
        self.mpc.update_world(world_config)

    def step(self) -> StepMpcResponse:
        """Step the MPC solver.

        Returns:
            The result of the step.
        """
        result = self.mpc.step(self.current_joint_state)
        return StepMpcResponse(
            solve_time=result.solve_time,
            action=JointStateModel(
                position=result.action.position.detach().cpu().numpy().tolist(),
                velocity=result.action.velocity.detach().cpu().numpy().tolist(),
                acceleration=result.action.acceleration.detach().cpu().numpy().tolist(),
                joint_names=result.action.joint_names,
                jerk=result.action.jerk.detach().cpu().numpy().tolist(),
            ),
            metrics=ArmReacherMetricsModel(
                cost=result.metrics.cost.detach().cpu().numpy().tolist(),
                constraint=result.metrics.constraint.detach().cpu().numpy().tolist(),
                feasible=result.metrics.feasible.detach().cpu().numpy().tolist(),
                position_error=result.metrics.position_error.detach().cpu().numpy().tolist(),
                rotation_error=result.metrics.rotation_error.detach().cpu().numpy().tolist(),
                pose_error=result.metrics.pose_error.detach().cpu().numpy().tolist(),
                goalset_index=result.metrics.goalset_index.detach().cpu().numpy().tolist(),
                null_space_error=result.metrics.null_space_error.detach().cpu().numpy().tolist(),
            ),
        )
