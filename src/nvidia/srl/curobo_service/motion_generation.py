# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Motion generation functions."""

# Standard Library
import functools
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Third Party
import torch

# cuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

# NVIDIA
from nvidia.srl.curobo_service.data_models import (
    GraspPlanRequest,
    GraspPlanResponse,
    JointMotionGenPlanRequest,
    JointStateTrajectory,
    MotionGenPlanRequest,
    MotionPlanResponse,
)
from nvidia.srl.curobo_service.utils import (
    convert_tensor_to_float,
    convert_tensor_to_int,
    convert_tensors_to_list_of_lists,
    create_world_config,
    perf_timer,
    robot_config_from_request,
)
from nvidia.srl.tools.logger import DEBUG, Logger

logger = Logger(name=Path(__file__).stem, log_level=DEBUG)


def hash_motion_plan_request_for_motion_gen(
    request: Union[MotionGenPlanRequest, JointMotionGenPlanRequest],
) -> int:
    """Hash motion plan request for motion gen object retrieval.

    Note this is NOT hashing all fields in the entire request, just the subset
    required to build and warmup the motiongen object.

    Args:
        request: MotionGenPlanRequest

    Returns:
        Hash value
    """
    return hash(
        (
            request.yaml,
            request.urdf,
            request.base_link,
            request.ee_link,
            request.world_config,
        )
    )


def hash_mesh_path_list(mesh_path_list: Optional[List[str]]) -> int:
    """Hash mesh path list. Read the contents of the files and hash them."""
    if mesh_path_list is None:
        return 0
    return hash(tuple(open(p, "rb").read() for p in mesh_path_list))


class MotionGenPlanRequestWithCustomHash:
    """Adds hashing to MotionGenPlanRequest to allow caching of MotionGen object."""

    def __init__(
        self,
        request: Union[MotionGenPlanRequest, JointMotionGenPlanRequest],
        mesh_path_list: Optional[List[str]] = None,
    ) -> None:
        """Initialize MotionGenPlanRequestWithCustomHash.

        Args:
            request: Request
            mesh_path_list: List of mesh file paths that form the world
        """
        self.request = request
        self.mesh_path_list = mesh_path_list

        # Compute hash once during initialization when files are guaranteed to exist
        self._hash_value = hash(
            (
                hash_motion_plan_request_for_motion_gen(self.request),
                hash_mesh_path_list(self.mesh_path_list),
            )
        )

    def __hash__(self) -> int:
        """Hash MotionGenPlanRequestWithCustomHash."""
        return self._hash_value

    def hash(self) -> int:
        """Hash MotionGenPlanRequestWithCustomHash."""
        return self._hash_value

    def __eq__(self, other: object) -> bool:
        """Equality operator."""
        if not isinstance(other, MotionGenPlanRequestWithCustomHash):
            return False
        return self.hash() == other.hash()


@perf_timer(custom_print=logger.debug)
def get_motion_gen(
    request: Union[MotionGenPlanRequest, JointMotionGenPlanRequest],
    mesh_path_list: Optional[List[str]] = None,
) -> MotionGen:
    """Get motion generator.

    Args:
        request: Request
        mesh_path_list: List of mesh file paths that form the world

    Returns:get_motion_gen
        Motion generator
    """
    request_hashable = MotionGenPlanRequestWithCustomHash(request, mesh_path_list)
    return get_motion_gen_hashable_input(request_hashable)


@functools.lru_cache(maxsize=16)
def get_motion_gen_hashable_input(hashable_obj: MotionGenPlanRequestWithCustomHash) -> MotionGen:
    """This is a cacheable call to the construction and warmup of a MotionGen object."""
    tensor_args = TensorDeviceType()
    robot_cfg = robot_config_from_request(hashable_obj.request, tensor_args)
    world_cfg = create_world_config(hashable_obj.request.world_config)
    # Set up the motion generator
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        interpolation_dt=0.01,
        use_cuda_graph=True,
        tensor_args=tensor_args,
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup()
    return motion_gen


@perf_timer(custom_print=logger.debug)
def plan_single_motion_gen(
    motion_gen: MotionGen,
    start_state: List[float],
    goal_ee_position: List[float],
    goal_ee_quaternion: List[float],
    tensor_args: TensorDeviceType,
) -> MotionPlanResponse:
    """Runs the curobo plan single function.

    Args:
        motion_gen: Motion generator
        start_state: Start state
        goal_ee_position: Goal end-effector position
        goal_ee_quaternion: Goal end-effector quaternion
        tensor_args: Tensor device type
    """
    # Set up the problem
    goal_pose = Pose.from_list(goal_ee_position + goal_ee_quaternion)  # x, y, z, qw, qx, qy, qz
    start_joint_state = JointState.from_position(
        torch.tensor([[s for s in start_state]], **(tensor_args.as_torch_dict()))
    )

    # Plan
    result = motion_gen.plan_single(
        start_joint_state, goal_pose, MotionGenPlanConfig(max_attempts=1)
    )

    success_flag = result.success
    if success_flag is not None:
        success = success_flag.cpu().numpy()[0]
    else:
        success = False

    # Deal with status flag
    status_flag = result.status if result.status is not None else None
    if status_flag is not None:
        if isinstance(status_flag, str):
            status_value = status_flag
        else:
            status_value = status_flag.value
    else:
        status_value = None

    # Handle failure
    if not success:
        return MotionPlanResponse(
            success=success,
            status=status_value,
            dt=result.interpolation_dt,
            position=None,
            velocity=None,
            acceleration=None,
            jerk=None,
        )

    # Interpolate
    traj = result.get_interpolated_plan()  # result.interpolation_dt has the dt between timesteps

    # Convert to list of lists
    return MotionPlanResponse(
        success=success,
        status=status_value,
        dt=result.interpolation_dt,
        position=convert_tensors_to_list_of_lists(traj.position) if traj is not None else None,
        velocity=convert_tensors_to_list_of_lists(traj.velocity) if traj is not None else None,
        acceleration=(
            convert_tensors_to_list_of_lists(traj.acceleration) if traj is not None else None
        ),
        jerk=convert_tensors_to_list_of_lists(traj.jerk) if traj is not None else None,
    )


def plan_single_request(
    request: MotionGenPlanRequest, mesh_path_list: Optional[List[str]] = None
) -> MotionPlanResponse:
    """Plan motion.

    Args:
        request: Motion Generation Request

    Returns:
        Motion Plan Response.
    """
    motion_gen = get_motion_gen(request, mesh_path_list)
    tensor_args = TensorDeviceType()
    return plan_single_motion_gen(
        motion_gen,
        request.start_state,
        request.goal_ee_position,
        request.goal_ee_quaternion,
        tensor_args,
    )


@perf_timer(custom_print=logger.debug)
def plan_single_motion_gen_js(
    motion_gen: MotionGen,
    start_state: List[float],
    goal_state: List[float],
    tensor_args: TensorDeviceType,
) -> MotionPlanResponse:
    """Runs the curobo plan single js function.

    Args:
        motion_gen: Motion generator
        start_state: Start state
        goal_state: Goal state
        tensor_args: Tensor device type
    """
    # Set up the problem
    start_joint_state = JointState.from_position(
        torch.tensor([[s for s in start_state]], **(tensor_args.as_torch_dict()))
    )
    goal_joint_state = JointState.from_position(
        torch.tensor([[s for s in goal_state]], **(tensor_args.as_torch_dict()))
    )

    # Plan
    result = motion_gen.plan_single_js(start_joint_state, goal_joint_state)

    success_flag = result.success
    if success_flag is not None:
        success = success_flag.cpu().numpy()[0]
    else:
        success = False

    # Deal with status flag
    status_flag = result.status if result.status is not None else None
    if status_flag is not None:
        if isinstance(status_flag, str):
            status_value = status_flag
        else:
            status_value = status_flag.value
    else:
        status_value = None

    # Handle failure
    if not success:
        return MotionPlanResponse(
            success=success,
            status=status_value,
            dt=result.interpolation_dt,
            position=None,
            velocity=None,
            acceleration=None,
            jerk=None,
        )

    # Interpolate
    traj = result.get_interpolated_plan()  # result.interpolation_dt has the dt between timesteps

    # Convert to list of lists
    return MotionPlanResponse(
        success=success,
        status=status_value,
        dt=result.interpolation_dt,
        position=(convert_tensors_to_list_of_lists(traj.position) if traj is not None else None),
        velocity=(convert_tensors_to_list_of_lists(traj.velocity) if traj is not None else None),
        acceleration=(
            convert_tensors_to_list_of_lists(traj.acceleration) if traj is not None else None
        ),
        jerk=convert_tensors_to_list_of_lists(traj.jerk) if traj is not None else None,
    )


def plan_single_js_request(
    request: JointMotionGenPlanRequest, mesh_path_list: Optional[List[str]] = None
) -> MotionPlanResponse:
    """Plan joint motion.

    Args:
        request: Joint Motion Generation Request

    Returns:
        Joint Motion Plan Response.
    """
    motion_gen = get_motion_gen(request, mesh_path_list)
    tensor_args = TensorDeviceType()
    return plan_single_motion_gen_js(
        motion_gen,
        request.start_state,
        request.goal_state,
        tensor_args,
    )


def curobo_plan_grasp(
    motion_gen: MotionGen,
    start_state: List[float],
    grasp_poses: List[List[float]],
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
    """Plan grasp.

    Args:
        motion_gen: Motion generator
        start_state: Start state
        grasp_poses: Grasp poses [n_grasps, 7] where 7 is [x, y, z, qw, qx, qy, qz]
        disable_collision_links: List of links to disable collision during the final approach
        grasp_approach_offset: The offset of the grasp approach [x, y, z, qw, qx, qy, qz]
        retract_offset: The offset of the grasp retract [x, y, z, qw, qx, qy, qz]
        grasp_approach_path_constraint: The path constraint of the grasp approach
        retract_path_constraint: The path constraint of the grasp retract
        plan_approach_to_grasp: Plan approach to grasp
        plan_grasp_to_retract: Plan grasp to retract
        grasp_approach_constraint_in_goal_frame: Grasp approach constraint in goal frame
        retract_constraint_in_goal_frame: Retract constraint in goal frame
    """
    assert len(grasp_poses) > 0
    assert len(grasp_poses[0]) == 7
    tensor_args = TensorDeviceType()

    # Convert the start state to a tensor
    start_joint_state = JointState.from_position(
        torch.tensor([[s for s in start_state]], **(tensor_args.as_torch_dict()))
    )

    # Convert the grasp poses to a tensor
    grasp_poses_tensor = Pose(
        position=torch.tensor([[gp[:3] for gp in grasp_poses]], **(tensor_args.as_torch_dict())),
        quaternion=torch.tensor([[gp[3:] for gp in grasp_poses]], **(tensor_args.as_torch_dict())),
    )

    additional_kwargs: Dict[str, Any] = {}

    # Deal with the approach offset
    if grasp_approach_offset is not None:
        grasp_approach_offset_pose = Pose(
            position=torch.tensor(grasp_approach_offset[:3], **(tensor_args.as_torch_dict())),
            quaternion=torch.tensor(grasp_approach_offset[3:], **(tensor_args.as_torch_dict())),
        )
        additional_kwargs["grasp_approach_offset"] = grasp_approach_offset_pose

    # Deal with the retract offset
    if retract_offset is not None:
        retract_offset_pose = Pose(
            position=torch.tensor(retract_offset[:3], **(tensor_args.as_torch_dict())),
            quaternion=torch.tensor(retract_offset[3:], **(tensor_args.as_torch_dict())),
        )
        additional_kwargs["retract_offset"] = retract_offset_pose

    # Deal with the path constraints
    if retract_path_constraint is not None:
        assert len(retract_path_constraint) == 6
        additional_kwargs["retract_path_constraint"] = retract_path_constraint
    if grasp_approach_path_constraint is not None:
        assert len(grasp_approach_path_constraint) == 6
        additional_kwargs["grasp_approach_path_constraint"] = grasp_approach_path_constraint

    grasp_res = motion_gen.plan_grasp(
        start_joint_state,
        grasp_poses_tensor,
        MotionGenPlanConfig(max_attempts=1),
        disable_collision_links=disable_collision_links
        if disable_collision_links is not None
        else [],
        plan_approach_to_grasp=plan_approach_to_grasp,
        plan_grasp_to_retract=plan_grasp_to_retract,
        grasp_approach_constraint_in_goal_frame=grasp_approach_constraint_in_goal_frame,
        retract_constraint_in_goal_frame=retract_constraint_in_goal_frame,
        **additional_kwargs,
    )
    return GraspPlanResponse(
        success=convert_tensors_to_list_of_lists(grasp_res.success),
        grasp_trajectory=JointStateTrajectory(
            position=convert_tensors_to_list_of_lists(grasp_res.grasp_trajectory.position),
            velocity=convert_tensors_to_list_of_lists(grasp_res.grasp_trajectory.velocity),
            acceleration=convert_tensors_to_list_of_lists(grasp_res.grasp_trajectory.acceleration),
        )
        if grasp_res.grasp_trajectory is not None
        else None,
        grasp_trajectory_dt=convert_tensor_to_float(grasp_res.grasp_trajectory_dt),
        grasp_interpolated_trajectory=JointStateTrajectory(
            position=convert_tensors_to_list_of_lists(
                grasp_res.grasp_interpolated_trajectory.position
            ),
            velocity=convert_tensors_to_list_of_lists(
                grasp_res.grasp_interpolated_trajectory.velocity
            ),
            acceleration=convert_tensors_to_list_of_lists(
                grasp_res.grasp_interpolated_trajectory.acceleration
            ),
        )
        if grasp_res.grasp_interpolated_trajectory is not None
        else None,
        grasp_interpolation_dt=convert_tensor_to_float(grasp_res.grasp_interpolation_dt),
        retract_trajectory=JointStateTrajectory(
            position=convert_tensors_to_list_of_lists(grasp_res.retract_trajectory.position),
            velocity=convert_tensors_to_list_of_lists(grasp_res.retract_trajectory.velocity),
            acceleration=convert_tensors_to_list_of_lists(
                grasp_res.retract_trajectory.acceleration
            ),
        )
        if grasp_res.retract_trajectory is not None
        else None,
        retract_trajectory_dt=convert_tensor_to_float(grasp_res.retract_trajectory_dt),
        retract_interpolated_trajectory=JointStateTrajectory(
            position=convert_tensors_to_list_of_lists(
                grasp_res.retract_interpolated_trajectory.position
            ),
            velocity=convert_tensors_to_list_of_lists(
                grasp_res.retract_interpolated_trajectory.velocity
            ),
            acceleration=convert_tensors_to_list_of_lists(
                grasp_res.retract_interpolated_trajectory.acceleration
            ),
        )
        if grasp_res.retract_interpolated_trajectory is not None
        else None,
        retract_interpolation_dt=convert_tensor_to_float(grasp_res.retract_interpolation_dt),
        status=grasp_res.status if grasp_res.status is not None else None,
        planning_time=grasp_res.planning_time,
        goalset_index=convert_tensor_to_int(grasp_res.goalset_index)
        if grasp_res.goalset_index is not None
        else None,
    )


def plan_grasp_request(
    request: GraspPlanRequest, mesh_path_list: Optional[List[str]] = None
) -> GraspPlanResponse:
    """Plan grasp.

    Args:
        request: Grasp Generation Request

    Returns:
        Grasp Motion Plan Response.
    """
    motion_gen = get_motion_gen(request, mesh_path_list)
    return curobo_plan_grasp(
        motion_gen,
        request.start_state,
        request.grasp_poses,
        disable_collision_links=request.disable_collision_links,
        grasp_approach_offset=request.grasp_approach_offset,
        retract_offset=request.retract_offset,
        plan_approach_to_grasp=request.plan_approach_to_grasp,
        plan_grasp_to_retract=request.plan_grasp_to_retract,
        grasp_approach_constraint_in_goal_frame=request.grasp_approach_constraint_in_goal_frame,
        retract_constraint_in_goal_frame=request.retract_constraint_in_goal_frame,
        grasp_approach_path_constraint=request.grasp_approach_path_constraint,
        retract_path_constraint=request.retract_path_constraint,
    )
