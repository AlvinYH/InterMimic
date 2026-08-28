"""InterMimic Studio task for the shared passive articulated scene."""

from __future__ import annotations

import json
import os
from pathlib import Path

from isaacgym import gymapi, gymtorch
import numpy as np
import torch

from .intermimic import InterMimic, compute_sdf
from isaacgym.torch_utils import quat_mul, to_torch
from ...utils import torch_utils


def _active_link_contact_reward(
    *,
    intended,
    distance,
    hand_force,
    link_force,
    distance_threshold,
    force_threshold,
    missing_weight,
):
    """Reward intended hand contact only when it reaches the active link."""

    if intended.shape != hand_force.shape or intended.ndim != 2:
        raise ValueError("intended and hand_force must share shape (envs, hands)")
    if distance.ndim != 3 or distance.shape[:2] != intended.shape:
        raise ValueError("distance must have shape (envs, hands, active_links)")
    if link_force.shape != (intended.shape[0], distance.shape[2]):
        raise ValueError("link_force must have shape (envs, active_links)")

    intended_contact = intended > 0.1
    near_link = distance <= distance_threshold
    loaded_hand = hand_force >= force_threshold
    loaded_link = link_force[:, None, :] >= force_threshold
    live_contact = loaded_hand & torch.any(near_link & loaded_link, dim=2)
    nearest_distance = distance.amin(dim=2)

    floor = 0.5 * (
        1.0
        + torch.exp(
            torch.as_tensor(
                -float(missing_weight),
                dtype=nearest_distance.dtype,
                device=nearest_distance.device,
            )
        )
    )
    proximity = torch.exp(
        -nearest_distance.clamp_min(0.0) / float(distance_threshold)
    )
    proximity = torch.where(
        torch.isfinite(proximity), proximity, torch.zeros_like(proximity)
    )
    score = torch.where(
        live_contact,
        torch.ones_like(proximity),
        0.5 * proximity,
    )
    hand_reward = torch.where(
        intended_contact,
        floor + (1.0 - floor) * score,
        torch.ones_like(score),
    )
    hand_error = (intended_contact & ~live_contact).to(dtype=distance.dtype)
    return hand_reward, hand_error, live_contact


def _initial_object_reset_state(
    initial_qpos,
    env_ids,
):
    """Reset every object joint to the case-defined physical initial state."""

    qpos = initial_qpos.expand(len(env_ids), -1)
    return qpos, torch.zeros_like(qpos)


def _object_creation_pose(root_pos, root_rot):
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"object_root_pos must have shape (frames, 3), got {root_pos.shape}")
    if root_rot.shape != (root_pos.shape[0], 4):
        raise ValueError(
            "object_root_rot_xyzw must have shape (frames, 4), got "
            f"{root_rot.shape}"
        )
    if not np.all(np.isfinite(root_pos)) or not np.all(np.isfinite(root_rot)):
        raise ValueError("Object root reference must be finite")
    norm_error = np.max(np.abs(np.linalg.norm(root_rot, axis=1) - 1.0))
    if norm_error > 1e-4:
        raise ValueError(f"Object root quaternion norm error is too large: {norm_error}")
    return root_pos[0].copy(), root_rot[0].copy()


def _creation_dof_state(state, initial_qpos, initial_qvel):
    if state.shape != (len(initial_qpos),):
        raise ValueError(
            "Actor DOF state and reference-frame-0 qpos disagree: "
            f"{state.shape} vs {initial_qpos.shape}"
        )
    if initial_qvel.shape != initial_qpos.shape:
        raise ValueError("Reference-frame-0 qpos and qvel disagree")
    if state.dtype.names is None or not {"pos", "vel"}.issubset(state.dtype.names):
        raise ValueError("Actor DOF state must expose pos/vel fields")
    state = state.copy()
    state["pos"] = initial_qpos
    state["vel"] = initial_qvel
    return state


class InterMimicArticulated(InterMimic):
    """Keep native InterMimic PPO and add passive object q/link task state."""

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        global box_region_surface_distances
        global capsule_region_surface_distances
        global hand2_body_groups
        global load_hand_collision_geometry
        from pipeline.physics.common_rollout import SMPLX_BODY_NAMES, SMPLX_DOF_NAMES
        from pipeline.physics.contact import (
            box_region_surface_distances,
            capsule_region_surface_distances,
            hand2_body_groups,
            load_hand_collision_geometry,
        )
        from pipeline.physics.articulated_scene import (
            ARTICULATED_OBJECT_COLLISION_FILTER,
            STATIC_SCENE_COLLISION_FILTER,
            add_ground_plane,
            configure_articulated_actor,
            create_static_box_actors,
            load_articulated_asset,
            load_static_box_assets,
            validate_humanoid_object_collision_filters,
        )
        self._object_collision_filter = ARTICULATED_OBJECT_COLLISION_FILTER
        self._add_ground_plane = add_ground_plane
        self._validate_humanoid_object_collision_filters = (
            validate_humanoid_object_collision_filters
        )

        env = cfg["env"]
        manifest_path = Path(env["articulatedInputPath"]).expanduser().resolve()
        self._object_config = json.loads(manifest_path.read_text(encoding="utf-8"))
        static_collision_filter = self._object_config.get(
            "static_box_collision_filter",
            STATIC_SCENE_COLLISION_FILTER,
        )
        if (
            isinstance(static_collision_filter, bool)
            or not isinstance(static_collision_filter, int)
            or static_collision_filter < 0
        ):
            raise ValueError("static_box_collision_filter must be a non-negative integer")
        self._static_collision_filter = static_collision_filter
        object_reference_path = Path(
            self._object_config["reference_path"]
        ).expanduser().resolve()
        with np.load(object_reference_path, allow_pickle=False) as values:
            object_root_pos = np.asarray(values["object_root_pos"], dtype=np.float32)
            object_root_rot = np.asarray(values["object_root_rot_xyzw"], dtype=np.float32)
            object_root_vel = np.asarray(
                values["object_root_vel"], dtype=np.float32
            )
            object_root_ang_vel = np.asarray(
                values["object_root_ang_vel"], dtype=np.float32
            )
            self._root_reference_np = np.concatenate(
                (
                    object_root_pos,
                    object_root_rot,
                    object_root_vel,
                    object_root_ang_vel,
                ),
                axis=1,
            )
            self._q_reference_np = np.asarray(values["object_joint_qpos"], dtype=np.float32)
            self._qvel_reference_np = np.asarray(
                values["object_joint_qvel"], dtype=np.float32
            )
            self._joint_types = [
                str(value) for value in np.asarray(values["joint_types"]).tolist()
            ]
            reference_joint_names = [
                str(value) for value in np.asarray(values["joint_names"]).tolist()
            ]
            reference_active_joint_names = [
                str(value)
                for value in np.asarray(values["active_joint_names"]).tolist()
            ]
            reference_active_parent_names = [
                str(value)
                for value in np.asarray(values["active_parent_link_names"]).tolist()
            ]
            reference_active_child_names = [
                str(value)
                for value in np.asarray(values["active_child_link_names"]).tolist()
            ]
            self._link_reference_np = np.asarray(values["object_link_pos"], dtype=np.float32)
            self._link_reference_rot_np = np.asarray(
                values["object_link_rot_xyzw"], dtype=np.float32
            )
            self._link_reference_vel_np = np.asarray(
                values["object_link_vel"], dtype=np.float32
            )
            self._link_reference_ang_vel_np = np.asarray(
                values["object_link_ang_vel"], dtype=np.float32
            )
            self._reference_link_names = [
                str(value) for value in np.asarray(values["body_names"]).tolist()
            ]
            self._intended_contact_np = np.asarray(values["intended"], dtype=np.bool_)
            self._reference_fps = float(np.asarray(values["fps"]).item())
            self._object_surface_mode = str(
                np.asarray(values["object_surface_mode"]).item()
            )
            self._object_surface_points_np = np.asarray(
                values["object_surface_points_link_local_scaled"], dtype=np.float32
            )
            self._object_surface_link_names = [
                str(value)
                for value in np.asarray(
                    values["object_surface_point_link_names"]
                ).tolist()
            ]
            self._contact_points_np = np.asarray(
                values["collision_surface_points_link_local_scaled"], dtype=np.float32
            )
            self._contact_point_link_names = [
                str(value)
                for value in np.asarray(
                    values["collision_surface_point_link_names"]
                ).tolist()
            ]
        self._object_creation_pos_np, self._object_creation_rot_np = _object_creation_pose(
            object_root_pos, object_root_rot
        )

        self._joint_names = [str(value) for value in self._object_config["joint_names"]]
        self._active_joint_names = [
            str(value) for value in self._object_config["active_joint_names"]
        ]
        self._initial_qpos_np = np.asarray(
            self._object_config["initial_joint_qpos"],
            dtype=np.float32,
        ).reshape(-1)
        if (
            self._initial_qpos_np.shape != (len(self._joint_names),)
            or not np.isfinite(self._initial_qpos_np).all()
        ):
            raise ValueError("initial_joint_qpos must be finite and match object joints")
        self._initial_qvel_np = np.zeros_like(self._initial_qpos_np)
        self._object_dof_count = len(self._joint_names)
        self._active_link_names = [
            str(value)
            for value in self._object_config["active_child_link_names"]
        ]
        if len(self._active_link_names) != 1:
            raise ValueError(
                "InterMimic's fixed-width 21-D object tracking supports exactly "
                "one active task link; the shared scene still simulates every joint"
            )
        if (
            reference_joint_names != self._joint_names
            or self._joint_types
            != [str(value) for value in self._object_config["joint_types"]]
            or reference_active_joint_names != self._active_joint_names
            or reference_active_parent_names
            != [
                str(value)
                for value in self._object_config["active_parent_link_names"]
            ]
            or reference_active_child_names != self._active_link_names
            or self._reference_link_names
            != [str(value) for value in self._object_config["body_names"]]
        ):
            raise ValueError("Articulated manifest and reference topology disagree")
        if (
            self._q_reference_np.ndim != 2
            or self._q_reference_np.shape[1] != self._object_dof_count
        ):
            raise ValueError(
                "object_joint_qpos must have shape (frames, object DOFs), got "
                f"{self._q_reference_np.shape} for {self._object_dof_count} DOFs"
            )
        if self._link_reference_np.shape != (
            self._q_reference_np.shape[0],
            len(self._reference_link_names),
            3,
        ):
            raise ValueError(
                "object_link_pos must have shape (frames, links, 3), got "
                f"{self._link_reference_np.shape}"
            )
        if self._link_reference_rot_np.shape != (
            self._q_reference_np.shape[0],
            len(self._reference_link_names),
            4,
        ):
            raise ValueError(
                "object_link_rot_xyzw must have shape (frames, links, 4), got "
                f"{self._link_reference_rot_np.shape}"
            )
        expected_link_velocity_shape = self._link_reference_np.shape
        if (
            self._link_reference_vel_np.shape != expected_link_velocity_shape
            or self._link_reference_ang_vel_np.shape != expected_link_velocity_shape
            or not np.isfinite(self._link_reference_vel_np).all()
            or not np.isfinite(self._link_reference_ang_vel_np).all()
        ):
            raise ValueError("Object link velocity references must match link positions")
        if not np.all(np.isfinite(self._q_reference_np)):
            raise ValueError("Object q reference must be finite")
        if (
            self._root_reference_np.shape != (self._q_reference_np.shape[0], 13)
            or not np.isfinite(self._root_reference_np).all()
        ):
            raise ValueError("Object root reference must be finite and match the joint timeline")
        if (
            self._qvel_reference_np.shape != self._q_reference_np.shape
            or not np.all(np.isfinite(self._qvel_reference_np))
        ):
            raise ValueError("Object qvel reference must match object q reference")
        if self._intended_contact_np.shape != (self._q_reference_np.shape[0], 2):
            raise ValueError("intended contact must have shape (frames, 2)")
        if self._contact_points_np.shape != (len(self._contact_point_link_names), 3):
            raise ValueError("Collision-surface points and point-link names disagree")
        if not self._contact_point_link_names:
            raise ValueError("Object collision surface must not be empty")
        if self._object_surface_mode not in ("active_part", "full_object"):
            raise ValueError("Object surface mode must be active_part or full_object")
        if (
            self._object_surface_points_np.shape != (1024, 3)
            or len(self._object_surface_link_names) != 1024
        ):
            raise ValueError("Object interaction surface must contain exactly 1,024 points")
        if not np.isfinite(self._object_surface_points_np).all():
            raise ValueError("Object interaction surface must be finite")
        self._configure_articulated_actor = configure_articulated_actor
        self._create_static_box_actors = create_static_box_actors
        self._load_articulated_asset = load_articulated_asset
        self._load_static_box_assets = load_static_box_assets
        self._common_human_body_names = SMPLX_BODY_NAMES
        self._common_human_dof_names = SMPLX_DOF_NAMES

        self._target_contact_distance_threshold = float(
            env["articulationContactDistanceThreshold"]
        )
        self._target_contact_force_threshold = float(
            env["articulationContactForceThreshold"]
        )
        self._contact_distance_chunk_size = 64
        if (
            not np.isfinite(self._target_contact_distance_threshold)
            or self._target_contact_distance_threshold <= 0.0
        ):
            raise ValueError("articulationContactDistanceThreshold must be positive finite")
        if (
            not np.isfinite(self._target_contact_force_threshold)
            or self._target_contact_force_threshold <= 0.0
        ):
            raise ValueError("articulationContactForceThreshold must be positive finite")
        self._rollout_path = env["commonRolloutOutputPath"]
        self._rollout_fixed_horizon = bool(env.get("fixedHorizonRollout", False))
        self._rollout_fps = self._reference_fps
        if self._rollout_fps <= 0.0:
            raise ValueError("articulated reference FPS must be positive")
        if not np.isclose(self._rollout_fps, float(env["dataFPS"])):
            raise ValueError("OMOMO dataFPS and articulated reference FPS differ")
        if not np.isclose(
            float(env["plane"]["height"]),
            float(self._object_config["ground_height"]),
        ):
            raise ValueError("author ground plane and articulated manifest differ")
        self._humanoid_mjcf_path = Path(
            env["articulatedHumanoidXmlPath"]
        ).expanduser().resolve()
        self._rollout_frames = []
        self._rollout_reference_body_position = None
        self._rollout_next_frame = None
        self._rollout_written = False
        self._rollout_terminated = None
        self._contact_measurement_cache = None
        self._rollout_sanity_check = (
            os.environ.get("INTERMIMIC_ROLLOUT_SANITY_CHECK") == "1"
        )
        self._rollout_legacy_frames = []
        self._rollout_cached_contact_frames = 0
        self._q_reference = None
        self._static_box_handles = []
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        self._q_reference = torch.as_tensor(self._q_reference_np, device=self.device)
        self._qvel_reference = torch.as_tensor(
            self._qvel_reference_np,
            device=self.device,
        )
        self._root_reference = torch.as_tensor(
            self._root_reference_np,
            device=self.device,
        )
        self._link_reference = torch.as_tensor(self._link_reference_np, device=self.device)
        self._link_reference_rot = torch.as_tensor(
            self._link_reference_rot_np, device=self.device
        )
        self._link_reference_vel = torch.as_tensor(
            self._link_reference_vel_np, device=self.device
        )
        self._link_reference_ang_vel = torch.as_tensor(
            self._link_reference_ang_vel_np, device=self.device
        )
        self._intended_contact = torch.as_tensor(self._intended_contact_np, device=self.device)
        if self._rollout_path:
            self._rollout_terminated = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.bool
            )
        self._resolve_rollout_body_indices()
        self._previous_active_link_vel = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._previous_active_link_ang_vel = torch.zeros_like(
            self._previous_active_link_vel
        )
        if self._rollout_path:
            if self.num_envs != 1:
                raise ValueError("common rollout recording requires exactly one environment")
            reference_frames = torch.arange(
                self._q_reference_np.shape[0],
                device=self.device,
            )
            reference_ids = self.data_id[:1].expand_as(reference_frames)
            reference_body_pos = self.extract_data_component(
                "body_pos",
                ref=True,
                data_id=reference_ids,
                t=reference_frames,
            ).reshape(len(reference_frames), self.num_bodies, 3)
            self._rollout_reference_body_position = (
                reference_body_pos.detach().cpu().numpy().astype(np.float32)
            )
        from scripts.physics.capture_isaac_runtime import request_task_physics_dump
        request_task_physics_dump(self)

    def _load_target_asset(self):
        asset, properties = self._load_articulated_asset(
            self.gym,
            self.sim,
            self._object_config,
        )
        self._target_asset = [asset]
        self._target_dof_properties = properties
        self._static_box_assets = self._load_static_box_assets(
            self.gym,
            self.sim,
            self._object_config,
        )
        self._target_asset_body_names = list(self.gym.get_asset_rigid_body_names(asset))
        self.target_aggregate_body_capacity = (
            self.gym.get_asset_rigid_body_count(asset) + len(self._static_box_assets)
        )
        self.target_aggregate_shape_capacity = (
            self.gym.get_asset_rigid_shape_count(asset) + len(self._static_box_assets)
        )
        self.object_points = torch.as_tensor(
            self._object_surface_points_np[None],
            device=self.device,
        )

    def _create_ground_plane(self):
        self._add_ground_plane(self.gym, self.sim, self._object_config)

    def _build_target(self, env_id, env_ptr):
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*self._object_creation_pos_np)
        pose.r = gymapi.Quat(*self._object_creation_rot_np)
        handle = self.gym.create_actor(
            env_ptr,
            self._target_asset[0],
            pose,
            "articulated_object",
            env_id,
            self._object_collision_filter,
            0,
        )
        self._configure_articulated_actor(
            self.gym,
            env_ptr,
            handle,
            self._target_dof_properties,
            self._object_config,
        )
        creation_state = self.gym.get_actor_dof_states(
            env_ptr, handle, gymapi.STATE_ALL
        )
        creation_state = _creation_dof_state(
            creation_state,
            self._initial_qpos_np,
            self._initial_qvel_np,
        )
        self.gym.set_actor_dof_states(
            env_ptr, handle, creation_state, gymapi.STATE_ALL
        )
        self._target_handles.append(handle)

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        self._validate_humanoid_object_collision_filters(
            self.gym,
            env_ptr,
            self.humanoid_handles[env_id],
        )
        handles = self._create_static_box_actors(
            self.gym,
            env_ptr,
            env_id,
            self._static_box_assets,
            self._object_config,
            collision_filter=self._static_collision_filter,
        )
        if env_id == 0:
            self._static_box_handles = handles

    def _build_target_tensors(self):
        num_actors = self.get_num_actors_per_env()
        self._target_states = self._root_states.view(self.num_envs, num_actors, 13)[:, 1]
        self._tar_actor_ids = to_torch(
            num_actors * np.arange(self.num_envs) + 1, device=self.device, dtype=torch.int32
        )
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        all_dofs = self._dof_state.view(self.num_envs, dofs_per_env, 2)
        self._target_dof_pos = all_dofs[:, self.num_dof:self.num_dof + self._object_dof_count, 0]
        self._target_dof_vel = all_dofs[:, self.num_dof:self.num_dof + self._object_dof_count, 1]
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        all_bodies = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)
        count = len(self._target_asset_body_names)
        self._target_body_state = all_bodies[:, self.num_bodies:self.num_bodies + count]
        forces = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim))
        all_forces = forces.view(self.num_envs, bodies_per_env, 3)
        self._target_contact_forces = all_forces[:, self.num_bodies:self.num_bodies + count]
        self._tar_contact_forces = self._target_contact_forces.sum(dim=1)

    def _reset_target(self, env_ids):
        frames = torch.clamp(
            self.progress_buf[env_ids].long(),
            0,
            self._q_reference_np.shape[0] - 1,
        )
        self._target_states[env_ids] = self._root_reference[frames]
        if self._state_init in {
            InterMimic.StateInit.Default,
            InterMimic.StateInit.Start,
        }:
            reset_qpos, reset_qvel = _initial_object_reset_state(
                to_torch(self._initial_qpos_np, device=self.device),
                env_ids,
            )
        else:
            reset_qpos = self._q_reference[frames]
            reset_qvel = self._qvel_reference[frames]
        self._target_dof_pos[env_ids] = reset_qpos
        self._target_dof_vel[env_ids] = reset_qvel
        if hasattr(self, "_previous_active_link_vel"):
            self._previous_active_link_vel[env_ids] = self._link_reference_vel[
                frames, self._active_reference_link_id
            ]
            self._previous_active_link_ang_vel[env_ids] = (
                self._link_reference_ang_vel[frames, self._active_reference_link_id]
            )
        if self._rollout_terminated is not None:
            self._rollout_terminated[env_ids] = False

    def _reset_env_tensors(self, env_ids):
        human_ids = self._humanoid_actor_ids[env_ids]
        object_ids = self._tar_actor_ids[env_ids]
        actor_ids = torch.stack((human_ids, object_ids), dim=1).reshape(-1).contiguous()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_states), gymtorch.unwrap_tensor(actor_ids), len(actor_ids)
        )
        dof_actor_ids = actor_ids if self._object_dof_count else human_ids
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._dof_state), gymtorch.unwrap_tensor(dof_actor_ids), len(dof_actor_ids)
        )
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        from scripts.physics.capture_isaac_runtime import write_requested_task_physics
        write_requested_task_physics(self)
        if (
            not self._rollout_path
            or self._rollout_written
            or self.num_envs != 1
            or self._q_reference is None
            or self._rollout_next_frame is not None
            or not len(env_ids)
            or not torch.any(env_ids == 0)
        ):
            return
        if int(self.progress_buf[0].item()) != 0:
            raise RuntimeError("Formal articulated rollout must reset at reference frame 0")

        # This is the exact state written by the case-init frame-0 reset, before
        # the first physics step increments progress_buf. Rigid-body tensors are
        # only refreshed after simulation, so read the reset's canonical body
        # state from hoi_data while root/DOF/object state comes from the tensors
        # installed into Isaac Gym.
        data_id = self.data_id[:1]
        frame = self.progress_buf[:1]
        body_state = torch.cat(
            (
                self.extract_data_component("body_pos", ref=True, data_id=data_id, t=frame).reshape(self.num_bodies, 3),
                self.extract_data_component("body_rot", ref=True, data_id=data_id, t=frame).reshape(self.num_bodies, 4),
                self.extract_data_component("body_pos_vel", ref=True, data_id=data_id, t=frame).reshape(self.num_bodies, 3),
                self.extract_data_component("body_rot_vel", ref=True, data_id=data_id, t=frame).reshape(self.num_bodies, 3),
            ),
            dim=1,
        )
        link_count = len(self._target_contact_link_names)
        from pipeline.physics.contact import hand_region_distances

        reference_pos = self._link_reference[frame][
            :, self._contact_point_reference_link_ids
        ]
        reference_rot = self._link_reference_rot[frame][
            :, self._contact_point_reference_link_ids
        ]
        points = torch_utils.quat_rotate(
            reference_rot.reshape(-1, 4),
            self._contact_points.unsqueeze(0).reshape(-1, 3),
        ).view(1, -1, 3) + reference_pos
        distance = hand_region_distances(
            body_state[None, self._human_contact_body_ids],
            points,
            self._contact_point_link_ids,
            link_count,
            self._human_contact_capsule_endpoints,
            self._human_contact_capsule_radii,
            self._human_contact_capsule_valid,
            self._human_contact_box_centers,
            self._human_contact_box_quaternions,
            self._human_contact_box_half_extents,
            self._human_contact_box_valid,
            self._human_contact_local_groups,
        )[0]
        self._append_rollout_frame(
            human_root_state=self._humanoid_root_states[0],
            human_dof_pos=self._dof_pos[0],
            human_body_state=body_state,
            object_root_state=self._target_states[0],
            object_joint_qpos=self._target_dof_pos[0],
            region_distance_m=distance,
            hand_force_n=torch.zeros(2, dtype=torch.float32, device=self.device),
            region_force_n=torch.zeros(
                link_count, dtype=torch.float32, device=self.device
            ),
        )
        self._rollout_next_frame = 1

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        if self._pd_control:
            human = self._action_to_pd_targets(self.actions)
            targets = torch.cat(
                (human, torch.zeros_like(self._target_dof_pos)),
                dim=1,
            ).contiguous()
            self.gym.set_dof_position_target_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(targets),
                gymtorch.unwrap_tensor(self._humanoid_actor_ids),
                len(self._humanoid_actor_ids),
            )
        else:
            human = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
            forces = torch.cat((human, torch.zeros_like(self._target_dof_pos)), dim=1).contiguous()
            self.gym.set_dof_actuation_force_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(forces),
                gymtorch.unwrap_tensor(self._humanoid_actor_ids),
                len(self._humanoid_actor_ids),
            )

    def _reference_frame(self):
        return torch.clamp(self.progress_buf.long(), 0, self._q_reference.shape[0] - 1)

    def _compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = to_torch(
                np.arange(self.num_envs), device=self.device, dtype=torch.long
            )

        self._curr_ref_obs[env_ids] = self.hoi_data[
            self.data_id[env_ids], self.progress_buf[env_ids]
        ].clone()
        native = torch.cat(
            (
                self._compute_observations_iter(self.hoi_data, env_ids, 1),
                self._compute_observations_iter(self.hoi_data, env_ids, 16),
            ),
            dim=-1,
        )
        self.obs_buf[env_ids] = native

    def _compute_observations_iter(self, hoi_data, env_ids=None, delta_t=1):
        if env_ids is None:
            env_ids = to_torch(
                np.arange(self.num_envs), device=self.device, dtype=torch.long
            )

        next_ts = torch.clamp(
            self.progress_buf[env_ids] + delta_t,
            max=self.max_episode_length[self.data_id[env_ids]] - 1,
        )
        ref_obs = hoi_data[self.data_id[env_ids], next_ts].clone()
        obs = torch.cat(
            (
                self._compute_humanoid_obs(env_ids, ref_obs, next_ts),
                self._active_link_tracking_observation(env_ids, next_ts),
            ),
            dim=-1,
        )
        ig_all, ig, ref_ig = self._articulated_graph_observation(
            env_ids, ref_obs, next_ts
        )
        return torch.cat((obs, ig_all, ref_ig - ig), dim=-1)

    def _active_link_tracking_observation(self, env_ids, reference_frames):
        """Replace the author 21-D rigid-root feature with one active link."""

        root_states = self._humanoid_root_states[env_ids]
        root_pos = root_states[:, :3]
        root_rot = root_states[:, 3:7]
        live = self._target_body_state[env_ids, self._active_body_id]
        reference_pos = self._link_reference[
            reference_frames, self._active_reference_link_id
        ]
        reference_rot = self._link_reference_rot[
            reference_frames, self._active_reference_link_id
        ]
        reference_vel = self._link_reference_vel[
            reference_frames, self._active_reference_link_id
        ]
        reference_ang_vel = self._link_reference_ang_vel[
            reference_frames, self._active_reference_link_id
        ]

        heading = torch_utils.calc_heading_quat_inv(root_rot)
        heading_inv = torch_utils.calc_heading_quat(root_rot)
        local_vel = torch_utils.quat_rotate(heading, live[:, 7:10])
        local_ang_vel = torch_utils.quat_rotate(heading, live[:, 10:13])
        position_error = reference_pos - live[:, :3]
        local_position_error = torch_utils.quat_rotate(heading, position_error)

        rotation_error = torch_utils.quat_mul_norm(
            torch_utils.quat_inverse(reference_rot), live[:, 3:7]
        )
        local_rotation_error = quat_mul(
            quat_mul(heading, rotation_error), heading_inv
        )
        local_rotation_error = torch_utils.quat_to_tan_norm(local_rotation_error)
        local_velocity_error = torch_utils.quat_rotate(
            heading, reference_vel - live[:, 7:10]
        )
        local_angular_velocity_error = torch_utils.quat_rotate(
            heading, reference_ang_vel - live[:, 10:13]
        )
        return torch.cat(
            (
                local_vel,
                local_ang_vel,
                local_position_error,
                local_rotation_error,
                local_velocity_error,
                local_angular_velocity_error,
            ),
            dim=-1,
        )

    def _surface_points(self, state, local_points, body_ids):
        point_state = state[:, body_ids]
        points = local_points.unsqueeze(0).expand(len(state), -1, -1)
        return torch_utils.quat_rotate(
            point_state[..., 3:7].reshape(-1, 4), points.reshape(-1, 3)
        ).view(len(state), -1, 3) + point_state[..., :3]

    def _reference_surface_points(self, frames, local_points, reference_link_ids):
        positions = self._link_reference[frames][:, reference_link_ids]
        rotations = self._link_reference_rot[frames][:, reference_link_ids]
        points = local_points.unsqueeze(0).expand(len(frames), -1, -1)
        return torch_utils.quat_rotate(
            rotations.reshape(-1, 4), points.reshape(-1, 3)
        ).view(len(frames), -1, 3) + positions

    def _articulated_graph_observation(self, env_ids, ref_obs, next_ts):
        live_points = self._surface_points(
            self._target_body_state[env_ids],
            self._graph_surface_points,
            self._graph_body_ids,
        )
        ref_points = self._reference_surface_points(
            next_ts,
            self._graph_surface_points,
            self._graph_reference_link_ids,
        )

        live_ig = self._encode_graph(
            self._rigid_body_pos[env_ids],
            self._rigid_body_rot[env_ids, 0],
            live_points,
        )
        ref_ig = self._encode_graph(
            self.extract_data_component("body_pos", obs=ref_obs).view(
                len(env_ids), -1, 3
            ),
            self.extract_data_component("root_rot", obs=ref_obs),
            ref_points,
        )
        return (
            live_ig.reshape(len(env_ids), -1),
            live_ig[:, self._key_body_ids].reshape(len(env_ids), -1),
            ref_ig[:, self._key_body_ids].reshape(len(env_ids), -1),
        )

    def _encode_graph(self, body_pos, root_rot, object_points):
        graph = compute_sdf(body_pos, object_points)
        heading = torch_utils.calc_heading_quat_inv(root_rot)
        heading = heading.unsqueeze(1).expand(-1, body_pos.shape[1], -1)
        graph = torch_utils.quat_rotate(
            heading.reshape(-1, 4), graph.reshape(-1, 3)
        ).view_as(graph)
        norm = torch.linalg.norm(graph, dim=-1, keepdim=True)
        return graph / (norm + 1e-6) * torch.exp(-5.0 * norm)

    def _reference_human_interaction_graph(self):
        """Use the selected articulated surface in the unchanged human reward."""

        frames = self._reference_frame()
        body_pos = self.extract_data_component(
            "body_pos", obs=self._curr_ref_obs
        ).view(self.num_envs, -1, 3)
        object_points = self._reference_surface_points(
            frames,
            self._graph_surface_points,
            self._graph_reference_link_ids,
        )
        graph = compute_sdf(body_pos, object_points)
        heading = torch_utils.calc_heading_quat_inv(
            self.extract_data_component("root_rot", obs=self._curr_ref_obs)
        )
        return torch_utils.quat_rotate(
            heading.unsqueeze(1).expand(-1, body_pos.shape[1], -1).reshape(-1, 4),
            graph.reshape(-1, 3),
        ).view_as(graph)

    def compute_humanoid_reward(self, weights):
        """Keep the author formula, replacing only its rigid reference surface."""

        key_count = len(self._key_body_ids)
        key_pos = self.extract_data_component(
            "body_pos", obs=self._curr_obs
        ).view(self.num_envs, -1, 3)[:, self._key_body_ids]
        ref_key_pos = self.extract_data_component(
            "body_pos", obs=self._curr_ref_obs
        ).view(self.num_envs, -1, 3)[:, self._key_body_ids]
        reference_ig = self._reference_human_interaction_graph()
        weight_h = (-5.0 * reference_ig.norm(dim=-1)).exp()
        weight_hp = weight_h.clone().detach()
        ankle_toe_ids = [
            index
            for index in range(key_count)
            if "Ankle" in self.key_bodies[index] or "Toe" in self.key_bodies[index]
        ]
        weight_hp[:, ankle_toe_ids] = 1.0

        position_error = torch.mean(
            (ref_key_pos - key_pos).square().sum(dim=-1)
            * weight_hp[:, self._key_body_ids],
            dim=-1,
        )
        position_reward = torch.exp(-position_error * weights["p"])

        body_rot = self.extract_data_component(
            "body_rot", obs=self._curr_obs
        ).view(self.num_envs, -1, 4)
        reference_body_rot = self.extract_data_component(
            "body_rot", obs=self._curr_ref_obs
        ).view(self.num_envs, -1, 4)
        difference = torch_utils.quat_mul_norm(
            torch_utils.quat_inverse(reference_body_rot.reshape(-1, 4)),
            body_rot.reshape(-1, 4),
        )
        angle, _ = torch_utils.quat_to_angle_axis(difference)
        rotation_error = torch.mean(
            angle.view(-1, 52) * (1.0 - weight_h), dim=-1
        )
        rotation_reward = torch.exp(-rotation_error * weights["r"])

        body_velocity_error = torch.mean(
            (
                self.extract_data_component("body_pos_vel", obs=self._curr_ref_obs)
                - self.extract_data_component("body_pos_vel", obs=self._curr_obs)
            ).square(),
            dim=-1,
        )
        body_velocity_reward = torch.exp(-body_velocity_error * weights["pv"])
        rotation_velocity_error = torch.mean(
            (
                self.extract_data_component("body_rot_vel", obs=self._curr_ref_obs)
                - self.extract_data_component("body_rot_vel", obs=self._curr_obs)
            ).square(),
            dim=-1,
        )
        rotation_velocity_reward = torch.exp(
            -rotation_velocity_error * weights["rv"]
        )

        dof_acceleration = (
            self.extract_data_component("dof_vel", obs=self._curr_obs)
            - self.extract_data_component("dof_vel", obs=self._hist_obs)
        ) * self.fps_data
        dof_acceleration *= (
            self.progress_buf - self.start_times > 2
        ).float().unsqueeze(-1)
        energy_reward = torch.exp(
            -dof_acceleration.view(-1, 51 * 3).square().mean(dim=-1)
            * weights["eg1"]
        )
        reward = (
            position_reward
            * rotation_reward
            * body_velocity_reward
            * rotation_velocity_reward
            * energy_reward
        )
        reset = (ref_key_pos - key_pos).norm(dim=-1).mean(dim=-1) > 0.5
        return reward, reset, key_pos, ref_key_pos

    def _compute_reset(self):
        super()._compute_reset()
        if not self._rollout_path:
            return

        self._rollout_terminated |= self._terminate_buf.bool()
        self._terminate_buf[:] = self._rollout_terminated.to(self._terminate_buf.dtype)

    def compute_obj_reward(self, weights):
        """Use the author rigid-object reward on the active link instead."""

        self._contact_measurement_cache = None
        frames = self._reference_frame()
        live = self._target_body_state[:, self._active_body_id]
        reference_pos = self._link_reference[frames, self._active_reference_link_id]
        reference_rot = self._link_reference_rot[frames, self._active_reference_link_id]
        reference_vel = self._link_reference_vel[frames, self._active_reference_link_id]
        reference_ang_vel = self._link_reference_ang_vel[
            frames, self._active_reference_link_id
        ]

        root_pos = self.extract_data_component("root_pos", obs=self._curr_obs)
        root_rot = self.extract_data_component("root_rot", obs=self._curr_obs)
        heading = torch_utils.calc_heading_quat_inv(root_rot)
        local_pos = live[:, :3] - root_pos
        local_pos[..., -1] = live[:, 2]
        local_pos = torch_utils.quat_rotate(heading, local_pos)
        local_rot = quat_mul(heading, live[:, 3:7])

        reference_root_pos = self.extract_data_component(
            "root_pos", obs=self._curr_ref_obs
        )
        reference_root_rot = self.extract_data_component(
            "root_rot", obs=self._curr_ref_obs
        )
        reference_heading = torch_utils.calc_heading_quat_inv(reference_root_rot)
        reference_local_pos = reference_pos - reference_root_pos
        reference_local_pos[..., -1] = reference_pos[:, 2]
        reference_local_pos = torch_utils.quat_rotate(
            reference_heading, reference_local_pos
        )
        reference_local_rot = quat_mul(reference_heading, reference_rot)

        position_reward = torch.exp(
            -weights["op"]
            * torch.mean((reference_local_pos - local_pos) ** 2, dim=-1)
        )
        rotation_difference = torch_utils.quat_mul_norm(
            torch_utils.quat_inverse(reference_local_rot), local_rot
        )
        rotation_angle, _ = torch_utils.quat_to_angle_axis(rotation_difference)
        rotation_reward = torch.exp(-weights["or"] * rotation_angle)
        velocity_reward = torch.exp(
            -weights["opv"] * torch.mean((reference_vel - live[:, 7:10]) ** 2, dim=-1)
        )
        angular_velocity_reward = torch.exp(
            -weights["orv"]
            * torch.mean((reference_ang_vel - live[:, 10:13]) ** 2, dim=-1)
        )

        moving = (self.progress_buf - self.start_times > 2).float()
        linear_acceleration = (
            (live[:, 7:10] - self._previous_active_link_vel) * self.fps_data
        ) * moving.unsqueeze(-1)
        angular_acceleration = (
            (live[:, 10:13] - self._previous_active_link_ang_vel) * self.fps_data
        ) * moving.unsqueeze(-1)
        energy_reward = torch.exp(
            -weights["eg2"] * torch.mean(linear_acceleration.square(), dim=-1)
        ) * torch.exp(
            -weights["eg2"] * torch.mean(angular_acceleration.square(), dim=-1)
        )
        self._previous_active_link_vel.copy_(live[:, 7:10])
        self._previous_active_link_ang_vel.copy_(live[:, 10:13])

        obj_points = self._surface_points(
            self._target_body_state,
            self._object_surface_points,
            self._object_surface_body_ids,
        )
        ref_obj_points = self._reference_surface_points(
            frames,
            self._object_surface_points,
            self._object_surface_reference_link_ids,
        )
        object_reward = (
            position_reward
            * rotation_reward
            * velocity_reward
            * angular_velocity_reward
            * energy_reward
        )
        object_reset = (obj_points - ref_obj_points).norm(dim=-1).mean(dim=-1) > 0.5
        self.extras["active_link_position_reward"] = position_reward
        self.extras["active_link_rotation_reward"] = rotation_reward
        self.extras["active_link_velocity_reward"] = velocity_reward
        self.extras["active_link_angular_velocity_reward"] = angular_velocity_reward
        return object_reward, object_reset, obj_points, ref_obj_points

    def compute_ig_reward(self, weights, key_pos, ref_key_pos, _obj_points, _ref_obj_points):
        """Keep the author interaction-graph reward on the selected surface pool."""

        frames = self._reference_frame()
        obj_points = self._surface_points(
            self._target_body_state,
            self._graph_surface_points,
            self._graph_body_ids,
        )
        ref_obj_points = self._reference_surface_points(
            frames,
            self._graph_surface_points,
            self._graph_reference_link_ids,
        )
        key_count = len(self._key_body_ids)
        interaction = key_pos.view(-1, key_count, 3).unsqueeze(2) - obj_points.unsqueeze(1)
        reference_interaction = (
            ref_key_pos.view(-1, key_count, 3).unsqueeze(2) - ref_obj_points.unsqueeze(1)
        )
        weight = 1.0 / torch.clamp(interaction.square().sum(dim=-1), min=0.01)
        weight = weight / weight.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
        reference_weight = 1.0 / torch.clamp(
            reference_interaction.square().sum(dim=-1), min=0.01
        )
        reference_weight = reference_weight / reference_weight.sum(
            dim=-1, keepdim=True
        ).sum(dim=-2, keepdim=True)
        error = (interaction - reference_interaction).square().sum(dim=-1)
        reward = torch.exp(
            -weights["ig"] * (error * (weight + reference_weight)).sum(dim=(1, 2)) * 0.5
        )
        reference_normalized_error = (
            (interaction - reference_interaction).square().sum(dim=-1).sqrt()
            / torch.clamp(
                reference_interaction.square().sum(dim=-1).sqrt(), min=0.5
            )
        )
        live_normalized_error = (
            (interaction - reference_interaction).square().sum(dim=-1).sqrt()
            / torch.clamp(interaction.square().sum(dim=-1).sqrt(), min=0.5)
        )
        reset = torch.logical_or(
            reference_normalized_error.amax(dim=(1, 2)) > 2,
            live_normalized_error.amax(dim=(1, 2)) > 2,
        )
        return reward, reset

    def compute_cg_reward(self, weights):
        """Adapt the author contact graph to the active articulated link."""

        contact_threshold = 0.1
        frames = self._reference_frame()
        cached = getattr(self, "_contact_measurement_cache", None)
        if cached is not None and torch.equal(cached[0], frames):
            _, intended, distance, hand_force, link_force = cached
        else:
            intended = (self._intended_contact[frames] > contact_threshold).float()
            link_count = len(self._target_contact_link_names)
            distance = torch.full(
                (self.num_envs, intended.shape[1], link_count),
                float("inf"),
                device=self.device,
            )
            hand_force = torch.zeros_like(intended)
            link_force = torch.zeros(
                (self.num_envs, link_count),
                device=self.device,
            )
            active_env_ids = torch.nonzero(
                torch.any(intended > contact_threshold, dim=1),
                as_tuple=False,
            ).reshape(-1)
            if active_env_ids.numel() > 0:
                active_distance, active_hand_force, active_link_force = (
                    self._measure_contacts(active_env_ids)
                )
                distance[active_env_ids] = active_distance
                hand_force[active_env_ids] = active_hand_force
                link_force[active_env_ids] = active_link_force
            if self._rollout_path:
                # The recorder runs immediately after this reward calculation.
                # Keep the live CUDA tensors so it can reuse exactly these
                # contact measurements instead of issuing an identical second
                # geometry query for the same physics frame.
                self._contact_measurement_cache = (
                    frames.detach().clone(),
                    intended,
                    distance,
                    hand_force,
                    link_force,
                )

        hand_reward, hand_error, live_contact = _active_link_contact_reward(
            intended=intended,
            distance=distance,
            hand_force=hand_force,
            link_force=link_force,
            distance_threshold=self._target_contact_distance_threshold,
            force_threshold=self._target_contact_force_threshold,
            missing_weight=weights["cg_hand"],
        )
        human_contact = self.extract_data_component(
            "contact_human", obs=self._curr_obs
        )
        ref_other_contact = self.extract_data_component(
            "contact_human", obs=self._curr_ref_obs
        )[:, self._human_other_contact_body_ids]
        other_contact = human_contact[:, self._human_other_contact_body_ids]
        other_error = (
            torch.abs(other_contact - ref_other_contact)
            * (ref_other_contact > contact_threshold)
        ).mean(dim=1)
        other_reward = torch.exp(-other_error * weights["cg_other"])

        no_contact = torch.abs(other_contact) < contact_threshold
        prohibited_error = (
            torch.abs(no_contact + ref_other_contact)
            * (ref_other_contact < -contact_threshold)
        ).mean(dim=1)
        prohibited_reward = torch.exp(-prohibited_error * weights["cg_all"])

        contact_energy = self._contact_forces.abs().sum(dim=-1).sum(dim=-1)
        energy_reward = torch.exp(-contact_energy.pow(2) * weights["eg3"])
        reward = (
            hand_reward.prod(dim=1)
            * other_reward
            * prohibited_reward
            * energy_reward
        )
        self.extras["active_link_contact_reward"] = hand_reward.mean(dim=1)
        self.extras["active_link_contact_live"] = live_contact.float().mean(dim=1)
        nearest_distance = distance.amin(dim=2)
        self.extras["active_link_contact_distance_m"] = torch.where(
            torch.isfinite(nearest_distance),
            nearest_distance,
            torch.zeros_like(nearest_distance),
        ).mean(dim=1)
        return reward, hand_error

    def _resolve_rollout_body_indices(self):
        body_lookup = {name: i for i, name in enumerate(self._target_asset_body_names)}
        missing_links = [name for name in self._reference_link_names if name not in body_lookup]
        if missing_links:
            raise ValueError(f"Object reference links are absent from the loaded URDF: {missing_links}")
        reference_lookup = {
            name: index for index, name in enumerate(self._reference_link_names)
        }
        active_link = self._active_link_names[0]
        self._active_body_id = body_lookup[active_link]
        self._active_reference_link_id = reference_lookup[active_link]
        surface_links = set(self._object_surface_link_names)
        if self._object_surface_mode == "active_part":
            if surface_links != {active_link}:
                raise ValueError(
                    "active_part interaction surface must contain only the active task link"
                )
        elif not surface_links.issubset(set(self._reference_link_names)):
            raise ValueError("full_object interaction surface has unknown links")

        def surface_tensors(points, names, label):
            unknown = sorted(set(names).difference(body_lookup))
            if unknown:
                raise ValueError(f"{label} interaction surface has unknown links: {unknown}")
            return (
                torch.as_tensor(points, device=self.device, dtype=torch.float32),
                torch.as_tensor(
                    [body_lookup[name] for name in names],
                    device=self.device,
                    dtype=torch.long,
                ),
                torch.as_tensor(
                    [reference_lookup[name] for name in names],
                    device=self.device,
                    dtype=torch.long,
                ),
            )

        (
            self._object_surface_points,
            self._object_surface_body_ids,
            self._object_surface_reference_link_ids,
        ) = surface_tensors(
            self._object_surface_points_np,
            self._object_surface_link_names,
            self._object_surface_mode,
        )
        (
            self._graph_surface_points,
            self._graph_body_ids,
            self._graph_reference_link_ids,
        ) = (
            self._object_surface_points,
            self._object_surface_body_ids,
            self._object_surface_reference_link_ids,
        )

        human_names = list(self.gym.get_actor_rigid_body_names(self.envs[0], self.humanoid_handles[0]))
        if tuple(human_names) != self._common_human_body_names:
            raise ValueError("Loaded humanoid bodies do not match the common 52-body order")
        self._human_contact_body_groups = hand2_body_groups(human_names)
        flat_body_ids = tuple(
            body_id for group in self._human_contact_body_groups for body_id in group
        )
        hand_body_ids = set(flat_body_ids)
        self._human_other_contact_body_ids = tuple(
            body_id
            for body_id in range(len(self.contact_bodies))
            if body_id not in hand_body_ids
        )
        (
            capsule_endpoints,
            capsule_radii,
            capsule_valid,
            box_centers,
            box_quaternions,
            box_half_extents,
            box_valid,
        ) = load_hand_collision_geometry(
            self._humanoid_mjcf_path,
            [human_names[body_id] for body_id in flat_body_ids],
        )
        if not bool(np.logical_or(capsule_valid, box_valid).all()):
            missing = [
                name
                for name, has_capsule, has_box in zip(
                    [human_names[body_id] for body_id in flat_body_ids],
                    capsule_valid.tolist(),
                    box_valid.tolist(),
                )
                if not has_capsule and not has_box
            ]
            raise ValueError(f"Hand collision bodies have no capsule or box geometry: {missing}")
        self._human_contact_body_ids = torch.as_tensor(
            flat_body_ids, device=self.device, dtype=torch.long
        )
        self._human_contact_group_slices = (
            slice(0, len(self._human_contact_body_groups[0])),
            slice(len(self._human_contact_body_groups[0]), len(flat_body_ids)),
        )
        local_body_ids = torch.arange(
            len(flat_body_ids),
            device=self.device,
            dtype=torch.long,
        )
        self._human_contact_local_groups = tuple(
            local_body_ids[group]
            for group in self._human_contact_group_slices
        )
        self._human_contact_capsule_endpoints = torch.as_tensor(
            capsule_endpoints, device=self.device
        )
        self._human_contact_capsule_radii = torch.as_tensor(
            capsule_radii, device=self.device
        )
        self._human_contact_capsule_valid = torch.as_tensor(
            capsule_valid, device=self.device
        )
        self._human_contact_box_centers = torch.as_tensor(
            box_centers, device=self.device
        )
        self._human_contact_box_quaternions = torch.as_tensor(
            box_quaternions, device=self.device
        )
        self._human_contact_box_half_extents = torch.as_tensor(
            box_half_extents, device=self.device
        )
        self._human_contact_box_valid = torch.as_tensor(
            box_valid, device=self.device
        )

        self._target_contact_link_names = tuple(
            dict.fromkeys(self._contact_point_link_names)
        )
        target_contact_links = set(self._target_contact_link_names)
        if self._object_surface_mode == "active_part":
            if target_contact_links != {active_link}:
                raise ValueError(
                    "active_part collision surface must belong only to the active task link"
                )
        elif not target_contact_links.issubset(set(self._reference_link_names)):
            raise ValueError("full_object collision surface has unknown links")
        region_names = set(self._target_contact_link_names)
        missing_region_links = sorted(region_names.difference(body_lookup))
        if missing_region_links:
            raise ValueError(f"Contact region links are absent from the loaded URDF: {missing_region_links}")
        self._target_contact_body_ids = torch.as_tensor(
            [body_lookup[name] for name in self._target_contact_link_names],
            device=self.device,
            dtype=torch.long,
        )

        missing_point_links = [
            name for name in self._contact_point_link_names if name not in body_lookup
        ]
        if missing_point_links:
            raise ValueError("A contact-region point refers to an unresolved object link")
        point_body_ids = [body_lookup[name] for name in self._contact_point_link_names]
        self._contact_points = torch.as_tensor(
            self._contact_points_np, device=self.device, dtype=torch.float32
        )
        self._contact_point_body_ids = torch.as_tensor(
            point_body_ids, device=self.device, dtype=torch.long
        )
        point_link_lookup = {
            name: index
            for index, name in enumerate(self._target_contact_link_names)
        }
        self._contact_point_link_ids = torch.as_tensor(
            [point_link_lookup[name] for name in self._contact_point_link_names],
            device=self.device,
            dtype=torch.long,
        )
        missing_reference_links = sorted(
            set(self._contact_point_link_names).difference(reference_lookup)
        )
        if missing_reference_links:
            raise ValueError(
                "Contact-region links are absent from object reference: "
                f"{missing_reference_links}"
            )
        self._contact_point_reference_link_ids = torch.as_tensor(
            [reference_lookup[name] for name in self._contact_point_link_names],
            device=self.device,
            dtype=torch.long,
        )

    def _measure_contacts(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        env_count = int(env_ids.numel())
        link_count = len(self._target_contact_link_names)
        distance = torch.full(
            (env_count, 2, link_count),
            float("inf"),
            device=self.device,
        )
        if self._contact_points.numel() > 0:
            state = self._target_body_state[env_ids][
                :, self._contact_point_body_ids
            ]
            points = torch_utils.quat_rotate(
                state[..., 3:7].reshape(-1, 4),
                self._contact_points.unsqueeze(0)
                .expand(env_count, -1, -1)
                .reshape(-1, 3),
            ).view(env_count, -1, 3) + state[..., :3]
            human_state = self._rigid_body_state.view(self.num_envs, -1, 13)[
                env_ids
            ][
                :, self._human_contact_body_ids
            ]
            for link_index in range(link_count):
                link_points = points[
                    :,
                    self._contact_point_link_ids == link_index,
                ]
                body_distance = torch.full(
                    (env_count, human_state.shape[1]),
                    float("inf"),
                    device=self.device,
                )
                for point_chunk in link_points.split(
                    self._contact_distance_chunk_size,
                    dim=1,
                ):
                    capsule_distance = capsule_region_surface_distances(
                        body_pos=human_state[..., :3],
                        body_quat_xyzw=human_state[..., 3:7],
                        endpoints_local=self._human_contact_capsule_endpoints,
                        radii=self._human_contact_capsule_radii,
                        valid=self._human_contact_capsule_valid,
                        region_points=point_chunk,
                    )
                    box_distance = box_region_surface_distances(
                        body_pos=human_state[..., :3],
                        body_quat_xyzw=human_state[..., 3:7],
                        centers_local=self._human_contact_box_centers,
                        quaternions_local_xyzw=self._human_contact_box_quaternions,
                        half_extents=self._human_contact_box_half_extents,
                        valid=self._human_contact_box_valid,
                        region_points=point_chunk,
                    )
                    body_distance = torch.minimum(
                        body_distance,
                        torch.minimum(capsule_distance, box_distance),
                    )
                for label, group in enumerate(
                    self._human_contact_group_slices
                ):
                    distance[:, label, link_index] = body_distance[
                        :, group
                    ].min(dim=1).values

        hand_force = torch.stack(
            tuple(
                torch.linalg.norm(
                    self._contact_forces[env_ids][:, group],
                    dim=-1,
                ).amax(dim=1)
                for group in self._human_contact_body_groups
            ),
            dim=1,
        )
        region_force = torch.linalg.norm(
            self._target_contact_forces[env_ids][
                :, self._target_contact_body_ids
            ],
            dim=-1,
        )
        return distance, hand_force, region_force

    def _measure_reference_contacts(self, frames):
        intended = (self._intended_contact[frames] > 0.1).float()
        link_count = len(self._target_contact_link_names)
        distance = torch.full(
            (self.num_envs, intended.shape[1], link_count),
            float("inf"),
            device=self.device,
        )
        hand_force = torch.zeros_like(intended)
        region_force = torch.zeros(
            (self.num_envs, link_count),
            device=self.device,
        )
        active_env_ids = torch.nonzero(
            torch.any(intended > 0.1, dim=1),
            as_tuple=False,
        ).reshape(-1)
        if active_env_ids.numel() > 0:
            active_distance, active_hand_force, active_region_force = (
                self._measure_contacts(active_env_ids)
            )
            distance[active_env_ids] = active_distance
            hand_force[active_env_ids] = active_hand_force
            region_force[active_env_ids] = active_region_force
        self._contact_measurement_cache = (
            frames.detach().clone(),
            intended,
            distance,
            hand_force,
            region_force,
        )
        return intended, distance, hand_force, region_force

    def _append_rollout_frame(self, **values):
        """Stage a formal rollout frame on GPU until the episode is complete.

        The legacy recorder immediately transferred every tensor to NumPy on
        every simulation step.  The resulting host synchronizations dominate
        one-environment ARCTIC replay.  Cloning on device preserves the exact
        frame values while deferring the seven device-to-host transfers to the
        final NPZ write.
        """

        if self._rollout_sanity_check:
            self._rollout_legacy_frames.append(
                {
                    name: value.detach().cpu().numpy().copy()
                    for name, value in values.items()
                }
            )
        self._rollout_frames.append(
            {name: value.detach().clone() for name, value in values.items()}
        )

    def _rollout_contact_measurements(self, frame):
        """Return exact live contact telemetry without recomputing reward work.

        ``compute_cg_reward`` runs in ``super().post_physics_step()`` and,
        whenever the reference requests contact, has already evaluated
        ``_measure_contacts`` for env 0.  That value covers every hand and
        articulated link, so it is exactly the value the old recorder computed
        again immediately afterwards.  Non-contact reference frames retain the
        old direct measurement, because their reward cache intentionally uses
        infinities/zeros instead of live telemetry.
        """

        cached = self._contact_measurement_cache
        if cached is not None and bool(self._intended_contact_np[frame].any()):
            _cached_frame, _intended, distance, hand_force, region_force = cached
            if self._rollout_sanity_check:
                direct = self._measure_contacts()
                for label, cached_value, direct_value in zip(
                    ("distance", "hand_force", "region_force"),
                    (distance, hand_force, region_force),
                    direct,
                ):
                    if not torch.equal(cached_value, direct_value):
                        raise RuntimeError(
                            "rollout contact cache differs from direct "
                            f"measurement at frame {frame}: {label}"
                        )
                self._rollout_cached_contact_frames += 1
            return distance, hand_force, region_force
        return self._measure_contacts()

    def post_physics_step(self):
        super().post_physics_step()
        if not self._rollout_path or self._rollout_written:
            return
        if self._rollout_next_frame is None:
            return
        if self._rollout_fixed_horizon:
            # The formal protocol neither terminates nor resets.  The CPU
            # counter is therefore the reference frame and avoids a per-step
            # CUDA-to-host scalar synchronization.
            frame = self._rollout_next_frame
            if self._rollout_sanity_check:
                actual_frame = int(self._reference_frame()[0].item())
                if actual_frame != frame:
                    raise RuntimeError(
                        "fixed-horizon recorder frame mismatch: "
                        f"expected {frame}, got {actual_frame}"
                    )
        else:
            frame = int(self._reference_frame()[0].item())
            if frame != self._rollout_next_frame:
                raise RuntimeError(
                    "InterMimic recorder expected reference frame "
                    f"{self._rollout_next_frame}, got {frame}"
                )
        distance, hand_force, region_force = self._rollout_contact_measurements(
            frame
        )
        self._append_rollout_frame(
            human_root_state=self._humanoid_root_states[0],
            human_dof_pos=self._dof_pos[0],
            human_body_state=self._rigid_body_state.view(
                self.num_envs, -1, 13
            )[0, :self.num_bodies],
            object_root_state=self._target_states[0],
            object_joint_qpos=self._target_dof_pos[0],
            region_distance_m=distance[0],
            hand_force_n=hand_force[0],
            region_force_n=region_force[0],
        )
        self._rollout_next_frame += 1
        done = (
            frame >= self._q_reference.shape[0] - 1
            if self._rollout_fixed_horizon
            else bool(self.reset_buf[0].item())
            or frame >= self._q_reference.shape[0] - 1
        )
        if done:
            self._write_common_rollout()
            self._rollout_written = True

    def _write_common_rollout(self):
        total_frames = len(self._q_reference_np)
        valid_frames = len(self._rollout_frames)
        if not 1 <= valid_frames <= total_frames:
            raise RuntimeError("InterMimic rollout frames do not match the reference")

        def stacked(name):
            values = torch.stack(
                [frame[name] for frame in self._rollout_frames]
            ).detach().cpu().numpy().astype(np.float32)
            if valid_frames < total_frames:
                values = np.concatenate(
                    (
                        values,
                        np.broadcast_to(
                            values[-1],
                            (total_frames - valid_frames, *values.shape[1:]),
                        ),
                    ),
                    axis=0,
                )
            return values

        def states(name):
            values = stacked(name)
            if valid_frames < total_frames:
                values[valid_frames:] = values[valid_frames - 1]
            return values

        def forces(name):
            values = stacked(name)
            if valid_frames < total_frames:
                values[valid_frames:] = 0.0
            return values

        state_names = (
            "human_root_state",
            "human_dof_pos",
            "human_body_state",
            "object_root_state",
            "object_joint_qpos",
            "region_distance_m",
        )
        force_names = ("hand_force_n", "region_force_n")
        state_values = {name: states(name) for name in state_names}
        force_values = {name: forces(name) for name in force_names}
        if self._rollout_sanity_check:
            def legacy_values(name, *, force):
                values = np.stack(
                    [frame[name] for frame in self._rollout_legacy_frames]
                ).astype(np.float32)
                if valid_frames < total_frames:
                    tail = (
                        np.zeros(
                            (total_frames - valid_frames, *values.shape[1:]),
                            dtype=np.float32,
                        )
                        if force
                        else np.broadcast_to(
                            values[-1],
                            (total_frames - valid_frames, *values.shape[1:]),
                        )
                    )
                    values = np.concatenate((values, tail), axis=0)
                return values

            for name, values in state_values.items():
                if not np.array_equal(values, legacy_values(name, force=False)):
                    raise RuntimeError(
                        f"GPU-staged rollout differs from legacy output: {name}"
                    )
            for name, values in force_values.items():
                if not np.array_equal(values, legacy_values(name, force=True)):
                    raise RuntimeError(
                        f"GPU-staged rollout differs from legacy output: {name}"
                    )

        np.savez_compressed(
            self._rollout_path,
            fps=np.asarray(self._rollout_fps, dtype=np.float32),
            valid_frame_count=np.asarray(valid_frames, dtype=np.int64),
            human_root_state=state_values["human_root_state"],
            human_dof_pos=state_values["human_dof_pos"],
            human_body_state=state_values["human_body_state"],
            human_body_position_reference=self._rollout_reference_body_position,
            human_body_names=np.asarray(self._common_human_body_names),
            human_dof_names=np.asarray(self._common_human_dof_names),
            object_root_state=state_values["object_root_state"],
            object_joint_qpos=state_values["object_joint_qpos"],
            object_joint_qpos_reference=self._q_reference_np.astype(np.float32),
            joint_names=np.asarray(self._joint_names),
            joint_types=np.asarray(self._joint_types),
            region_distance_m=state_values["region_distance_m"],
            intended=np.asarray(self._intended_contact_np, dtype=np.bool_),
            hand_force_n=force_values["hand_force_n"],
            region_force_n=force_values["region_force_n"],
            contact_region_link_names=np.asarray(self._target_contact_link_names),
        )
        if self._rollout_sanity_check:
            Path(self._rollout_path).with_name(
                "rollout_recorder_sanity.json"
            ).write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "contact_cache_bitwise_equal": True,
                        "gpu_staging_bitwise_equal": True,
                        "cached_contact_frames": self._rollout_cached_contact_frames,
                        "valid_frame_count": valid_frames,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )


__all__ = ["InterMimicArticulated"]
