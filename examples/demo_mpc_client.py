# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Demo for the MPC FastAPI server."""

# Standard Library
import json
import time
from typing import Any, Dict, List, Optional

# Third Party
import numpy as np
import requests

# NVIDIA
from nvidia.srl.curobo_service.mpc_data_models import JointStateModel, NewMpcResponse


class MpcApiClient:
    """A client class to interact with the MPC FastAPI server via HTTP requests."""

    def __init__(
        self,
        base_url: str = "http://localhost:10000",
        yaml_path: str = "tests/test_data/franka/franka.yml",
        urdf_path: str = "tests/test_data/franka/franka_panda.urdf",
        step_dt: float = 0.03,
        world_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the client with a base URL and default file paths.

        Args:
            base_url: The base URL of the FastAPI server
            yaml_path: Local path to the YAML file describing the robot.
            urdf_path: Local path to the URDF file for the robot.
            step_dt: The default MPC timestep.
            world_config: A dictionary describing the world environment.
        """
        self.base_url = base_url.rstrip("/")
        self.yaml_path = yaml_path
        self.urdf_path = urdf_path
        self.step_dt = step_dt
        # Provide a default if none is given.
        self.world_config = world_config
        # We'll store the mpc_id after we create it.
        self.mpc_id: Optional[int] = None
        self.joint_names: Optional[List[str]] = None

    def create_mpc(self) -> NewMpcResponse:
        """Create a new MPC manager session and store the mpc_id on the class.

        Returns:
            mpc_id: An integer identifying this MPC session.
        """
        # Read YAML and URDF from disk.
        with open(self.yaml_path, "r") as f:
            yaml_txt = f.read()
        with open(self.urdf_path, "r") as f:
            urdf_str = f.read()

        # Convert the world config dict to JSON string.
        world_config_str = json.dumps(self.world_config)

        # Construct payload
        url = f"{self.base_url}/mpc/new/"
        payload = {
            "yaml": yaml_txt,
            "urdf": urdf_str,
            "world_config": world_config_str,
            "step_dt": self.step_dt,
        }
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        self.mpc_id = data["mpc_id"]
        self.joint_names = data["joint_names"]
        if self.mpc_id is None:
            raise ValueError("Failed to create MPC session.")
        return NewMpcResponse(
            success=data["success"], mpc_id=self.mpc_id, joint_names=data["joint_names"]
        )

    def shutdown_mpc(self) -> Dict[str, Any]:
        """Shutdown the currently active MPC session."""
        if self.mpc_id is None:
            raise ValueError("No active MPC session to shut down.")

        url = f"{self.base_url}/mpc/{self.mpc_id}/shutdown/"
        resp = requests.post(url)
        resp.raise_for_status()
        data = resp.json()
        # Clear the cached mpc_id
        self.mpc_id = None
        return data

    def set_joint_state(
        self,
        joint_positions: List[float],
        joint_velocities: Optional[List[float]] = None,
        joint_accelerations: Optional[List[float]] = None,
        joint_jerks: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Set the current joint state for the active MPC session."""
        if self.mpc_id is None:
            raise ValueError("No active MPC session. Please call create_mpc() first.")
        if self.joint_names is None:
            raise ValueError("Failed to get joint names.")

        req = JointStateModel(
            position=joint_positions,
            velocity=joint_velocities
            if joint_velocities is not None
            else [0.0] * len(joint_positions),
            acceleration=joint_accelerations
            if joint_accelerations is not None
            else [0.0] * len(joint_positions),
            joint_names=self.joint_names,
            jerk=joint_jerks if joint_jerks is not None else [0.0] * len(joint_positions),
        )

        url = f"{self.base_url}/mpc/{self.mpc_id}/set_joint_state/"
        resp = requests.post(url, json=req.model_dump())
        resp.raise_for_status()
        return resp.json()

    def set_goal_pose(self, ee_position: List[float], ee_quaternion: List[float]) -> Dict[str, Any]:
        """Set the end-effector goal pose for the active MPC session."""
        if self.mpc_id is None:
            raise ValueError("No active MPC session. Please call create_mpc() first.")

        url = f"{self.base_url}/mpc/{self.mpc_id}/set_goal_pose/"
        payload = {
            "ee_position": ee_position,
            "ee_quaternion": ee_quaternion,
        }
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    def initialize_goal(
        self, start_state_list: List[float], ee_position: List[float], ee_quaternion: List[float]
    ) -> Dict[str, Any]:
        """Setup the goal for the active MPC session."""
        if self.mpc_id is None:
            raise ValueError("No active MPC session. Please call create_mpc() first.")

        url = f"{self.base_url}/mpc/{self.mpc_id}/initialize_goal/"
        payload = {
            "start_state_joint_position": start_state_list,
            "goal_position": ee_position,
            "goal_quaternion": ee_quaternion,
        }
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    def step_mpc(self) -> Dict[str, Any]:
        """Perform an MPC step and get the resulting action/metrics for the active session."""
        if self.mpc_id is None:
            raise ValueError("No active MPC session. Please call create_mpc() first.")

        url = f"{self.base_url}/mpc/{self.mpc_id}/step/"
        resp = requests.post(url)
        resp.raise_for_status()
        return resp.json()


if __name__ == "__main__":
    client = MpcApiClient(
        base_url="http://localhost:10000",
        yaml_path="tests/test_data/franka/franka.yml",
        urdf_path="tests/test_data/franka/franka_panda.urdf",
        step_dt=0.03,
        world_config={
            "cuboid": {
                "table": {"dims": [2, 2, 0.2], "pose": [0.4, 0.0, -0.1, 1, 0, 0, 0]},
                "cube_1": {"dims": [0.1, 0.1, 0.2], "pose": [0.4, 0.0, 0.5, 1, 0, 0, 0]},
            },
        },
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

    # 1) Create an MPC session
    mpc_acceptance = client.create_mpc()
    print(f"Created MPC session with mpc_id={mpc_acceptance.mpc_id}")
    print(f"Joint names: {mpc_acceptance.joint_names}")

    # 2) Set a goal pose
    ee_position = [0.1634387969970703, 0.29817265272140503, 0.8977956175804138]
    ee_quaternion = [
        0.44470980763435364,
        -0.47451046109199524,
        -0.7496849894523621,
        -0.12265995889902115,
    ]
    client.initialize_goal(FRANKA_PANDA_VALID_STARTING_POINT, ee_position, ee_quaternion)

    # 3) Set the initial joint state
    client.set_joint_state(joint_positions=FRANKA_PANDA_VALID_STARTING_POINT)

    # 4) Step the solver multiple times
    pose_error = []
    start_time = time.time()
    steps = 1000
    for i in range(steps):
        result = client.step_mpc()
        print(
            f"Step {i} => solve_time: {result['solve_time']}, "
            f"pose_error: {result['metrics']['pose_error']}"
        )
        pose_error.append(np.array(result["metrics"]["pose_error"]).flatten()[0])

        # Set the next joint state
        client.set_joint_state(
            joint_positions=result["action"]["position"],
            joint_velocities=result["action"]["velocity"],
            joint_accelerations=result["action"]["acceleration"],
            joint_jerks=result["action"]["jerk"],
        )
    end_time = time.time()
    print(
        f"Total run time: {end_time - start_time}, commands per second:"
        f" {steps / (end_time - start_time)}"
    )

    # 5) Shutdown the session
    shutdown_resp = client.shutdown_mpc()
    print("MPC session shut down:", shutdown_resp)

    # Third Party
    import matplotlib.pyplot as plt

    plt.figure()
    plt.title("Pose error")
    plt.xlabel("Step")
    plt.ylabel("Pose error")
    plt.plot(pose_error)
    plt.grid(True)
    plt.show()
