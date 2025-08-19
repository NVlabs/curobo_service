# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""NVIDIA Curobo Service API."""

# Standard Library
import os
import random
import tempfile
import traceback
import warnings
from typing import Dict, List, Optional

# Third Party
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

# NVIDIA
# NVIDIA - Common functionality
from nvidia.srl.curobo_service.data_models import (
    ForwardKinematicsRequest,
    GraspPlanRequest,
    GraspPlanResponse,
    HealthCheck,
    InverseKinematicsRequest,
    InverseKinematicsResponse,
    JointMotionGenPlanRequest,
    MotionGenPlanRequest,
    MotionPlanResponse,
    MpcStatus,
    RobotModelStateResponse,
)
from nvidia.srl.curobo_service.forward_kinematics import compute_forward_kinematics_request
from nvidia.srl.curobo_service.inverse_kinematics import compute_inverse_kinematics_request
from nvidia.srl.curobo_service.motion_generation import (
    plan_grasp_request,
    plan_single_js_request,
    plan_single_request,
)
from nvidia.srl.curobo_service.utils import modify_world_config_for_mesh

# Check for MPC enable flag at module load time.
# MPC endpoints (and related functionality) will only be registered if the
# environment variable NRM_CUROBO_ENABLE_MPC is truthy.
ENABLE_MPC = os.getenv("NRM_CUROBO_ENABLE_MPC", "false").lower() in ("true", "1", "yes")
print(f"NRM_CUROBO_ENABLE_MPC: {ENABLE_MPC}")

if ENABLE_MPC:
    # NVIDIA - MPC functionality
    # NVIDIA
    from nvidia.srl.curobo_service.mpc import MpcManager
    from nvidia.srl.curobo_service.mpc_data_models import (
        InitializeGoalRequest,
        JointStateModel,
        NewMpcRequest,
        NewMpcResponse,
        SetGoalPoseRequest,
        StepMpcResponse,
    )

# Application Constants
NRM_CUROBO_PORT_DEFAULT: int = 10000

# Create FastAPI application and template renderer
app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


# ------------------------------------------------------------------------------
# Global Exception Handler
# ------------------------------------------------------------------------------
@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a JSON response with error details and a stack trace on exception."""
    stack_trace = traceback.format_exc()  # Get the stack trace as a string
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "stack_trace": stack_trace,
        },
    )


# ------------------------------------------------------------------------------
# Motion Planning Endpoints
# ------------------------------------------------------------------------------
@app.get("/")
def read_root() -> Dict:
    """Return a default message."""
    return {"Hello": "World"}


@app.get("/health/")
async def health_check() -> HealthCheck:
    """Health check."""
    return HealthCheck(status="ok")


@app.post("/forward_kinematics/")
async def forward_kinematics(request: ForwardKinematicsRequest) -> RobotModelStateResponse:
    """Compute forward kinematics with URDF."""
    return compute_forward_kinematics_request(request)


@app.post("/inverse_kinematics/")
def inverse_kinematics(request: InverseKinematicsRequest) -> InverseKinematicsResponse:
    """Compute inverse kinematics."""
    return compute_inverse_kinematics_request(request)


@app.post("/motion_generation/")
def motion_generation(request: MotionGenPlanRequest) -> MotionPlanResponse:
    """Plan motion with URDF."""
    return plan_single_request(request)


@app.post("/plan_grasp/")
def plan_grasp(request: GraspPlanRequest) -> GraspPlanResponse:
    """Plan grasp with URDF."""
    return plan_grasp_request(request)


@app.post("/motion_generation_with_mesh/")
async def motion_generation_with_mesh(
    request: str = Form(...), mesh_files: List[UploadFile] = File(None)  # Accept as form data
) -> MotionPlanResponse:
    """Plan motion.

    Optionally accepts mesh files for collision checking.
    """
    # Parse the JSON string back into the request model
    request_data = MotionGenPlanRequest.model_validate_json(request)

    # Process uploaded mesh files if provided
    if mesh_files:
        with tempfile.TemporaryDirectory() as temp_dir:
            mesh_path_list = []
            for mesh_file in mesh_files:
                mesh_path = os.path.join(temp_dir, mesh_file.filename)  # type: ignore
                with open(mesh_path, "wb") as f:
                    f.write(await mesh_file.read())
                mesh_path_list.append(mesh_path)
            if request_data.world_config is not None:
                world_config = modify_world_config_for_mesh(request_data.world_config, temp_dir)
                request_data.world_config = world_config
            return plan_single_request(request_data, mesh_path_list)
    else:
        return plan_single_request(request_data)


@app.post("/motion_generation_js/")
def motion_generation_js(request: JointMotionGenPlanRequest) -> MotionPlanResponse:
    """Plan joint motion with URDF."""
    return plan_single_js_request(request)


@app.post("/motion_generation_js_with_mesh/")
async def motion_generation_js_with_mesh(
    request: str = Form(...), mesh_files: List[UploadFile] = File(None)  # Accept as form data
) -> MotionPlanResponse:
    """Plan joint motion.

    Optionally accepts mesh files for collision checking.
    """
    # Parse the JSON string back into the request model
    request_data = JointMotionGenPlanRequest.model_validate_json(request)

    # Process uploaded mesh files if provided
    if mesh_files:
        with tempfile.TemporaryDirectory() as temp_dir:
            mesh_path_list = []
            for mesh_file in mesh_files:
                mesh_path = os.path.join(temp_dir, mesh_file.filename)  # type: ignore
                with open(mesh_path, "wb") as f:
                    f.write(await mesh_file.read())
                mesh_path_list.append(mesh_path)
            if request_data.world_config is not None:
                world_config = modify_world_config_for_mesh(request_data.world_config, temp_dir)
                request_data.world_config = world_config
            return plan_single_js_request(request_data, mesh_path_list)
    else:
        return plan_single_js_request(request_data)


# ------------------------------------------------------------------------------
# Conditionally Registered MPC Endpoints
# ------------------------------------------------------------------------------
if ENABLE_MPC:
    # Dictionary to hold active MPC sessions
    _mpc_manager_dict: Dict[int, MpcManager] = {}

    def check_for_mpc_id(mpc_id: int) -> bool:
        """Check if the MPC manager exists for the given ID."""
        return mpc_id in _mpc_manager_dict

    def new_mpc_id(yaml_txt: str, urdf_str: str, seed: Optional[int] = None) -> int:
        """Generate a new MPC ID based on input texts and a random salt."""
        if seed is not None:
            random.seed(seed)
        salt = random.randint(0, 100000)
        return hash((yaml_txt, urdf_str, salt))

    @app.post("/mpc/new/")
    def mpc_new(request: NewMpcRequest) -> NewMpcResponse:
        """Create a new MPC session from YAML and URDF and return a session ID."""
        mpc_id = new_mpc_id(request.yaml, request.urdf)
        if check_for_mpc_id(mpc_id):
            raise HTTPException(
                status_code=500, detail=f"MPC session already exists with id {mpc_id}"
            )
        mpc_manager = MpcManager.from_yaml(
            yaml_txt=request.yaml,
            urdf_str=request.urdf,
            world_config_str=request.world_config,
            step_dt=request.step_dt,
        )
        _mpc_manager_dict[mpc_id] = mpc_manager
        return NewMpcResponse(success=True, mpc_id=mpc_id, joint_names=mpc_manager.mpc.joint_names)

    @app.post("/mpc/{mpc_id}/status/")
    def mpc_status(mpc_id: int) -> MpcStatus:
        """Check the status of an existing MPC session."""
        if not check_for_mpc_id(mpc_id):
            return MpcStatus(status="NOT_FOUND")
        return MpcStatus(status="ACTIVE")

    @app.post("/mpc/{mpc_id}/shutdown/")
    def mpc_shutdown(mpc_id: int) -> Dict:
        """Shutdown (delete) an existing MPC session."""
        if not check_for_mpc_id(mpc_id):
            raise HTTPException(status_code=404, detail=f"MPC session not found with id {mpc_id}")
        del _mpc_manager_dict[mpc_id]
        return {"status": "ok"}

    @app.post("/mpc/{mpc_id}/set_joint_state/")
    def mpc_set_joint_state(mpc_id: int, request: JointStateModel) -> Dict:
        """Update the current joint state of an MPC session."""
        if not check_for_mpc_id(mpc_id):
            raise HTTPException(status_code=404, detail=f"MPC session not found with id {mpc_id}")
        _mpc_manager_dict[mpc_id].update_current_joint_state_request(request)
        return {"status": "ok"}

    @app.post("/mpc/{mpc_id}/initialize_goal/")
    def mpc_initialize_goal(mpc_id: int, request: InitializeGoalRequest) -> Dict:
        """Initialize the goal for an MPC session using a starting state and target."""
        if not check_for_mpc_id(mpc_id):
            raise HTTPException(status_code=404, detail=f"MPC session not found with id {mpc_id}")
        _mpc_manager_dict[mpc_id].setup_goal_from_position_and_quaternion(
            start_state_list=request.start_state_joint_position,
            goal_position=request.goal_position,
            goal_quaternion=request.goal_quaternion,
            link_poses_names=request.extra_link_names,
            link_poses_goal_positions=request.extra_link_positions,
            link_poses_goal_quaternions=request.extra_link_quaternions,
        )
        return {"status": "ok"}

    @app.post("/mpc/{mpc_id}/set_goal_pose/")
    def mpc_set_goal_pose(mpc_id: int, request: SetGoalPoseRequest) -> Dict:
        """Update the goal pose of an MPC session."""
        if not check_for_mpc_id(mpc_id):
            raise HTTPException(status_code=404, detail=f"MPC session not found with id {mpc_id}")
        _mpc_manager_dict[mpc_id].update_goal_pose_from_position_and_quaternion(
            goal_position=request.ee_position,
            goal_quaternion=request.ee_quaternion,
            link_poses_names=request.extra_link_names,
            link_poses_goal_positions=request.extra_link_positions,
            link_poses_goal_quaternions=request.extra_link_quaternions,
        )
        return {"status": "ok"}

    @app.post("/mpc/{mpc_id}/step/")
    def mpc_step(mpc_id: int) -> StepMpcResponse:
        """Advance one step of the MPC solver."""
        if not check_for_mpc_id(mpc_id):
            raise HTTPException(status_code=404, detail=f"MPC session not found with id {mpc_id}")
        return _mpc_manager_dict[mpc_id].step()


# ------------------------------------------------------------------------------
# Utility Functions for Launch Configuration
# ------------------------------------------------------------------------------
def get_nrm_curobo_run_local() -> bool:
    """Determine whether to run the server locally based on an environment variable."""
    default_flag = False
    env_val = os.getenv("NRM_CUROBO_RUN_LOCAL", str(default_flag))
    try:
        return env_val.lower() in ("true", "1", "yes")
    except ValueError:
        warnings.warn(
            f"Invalid value for environment variable NRM_CUROBO_RUN_LOCAL: {env_val}, "
            f"defaulting to {default_flag}."
        )
        return default_flag


def get_nrm_curobo_port() -> int:
    """Retrieve the port number from the environment variable or use the default."""
    default_port_str = str(NRM_CUROBO_PORT_DEFAULT)
    port_str = os.getenv("NRM_CUROBO_PORT", default_port_str)
    try:
        return int(port_str)
    except ValueError:
        default_port = int(default_port_str)
        warnings.warn(
            f"Invalid value for environment variable NRM_CUROBO_PORT: {port_str}, "
            f"defaulting to {default_port}."
        )
        return default_port


# ------------------------------------------------------------------------------
# Application Startup
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    nrm_curobo_run_local = get_nrm_curobo_run_local()
    nrm_curobo_port = get_nrm_curobo_port()
    host_ip = "127.0.0.1" if nrm_curobo_run_local else "0.0.0.0"
    uvicorn.run(app, host=host_ip, port=nrm_curobo_port)
