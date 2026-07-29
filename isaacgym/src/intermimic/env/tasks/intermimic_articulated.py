"""InterMimic Studio task for the shared passive articulated scene."""

from __future__ import annotations

import json
from pathlib import Path

from isaacgym import gymapi, gymtorch
import numpy as np
import torch

from .intermimic import InterMimic, compute_sdf
from isaacgym.torch_utils import to_torch
from ...utils import torch_utils


def _target_region_contact_reward(
    *,
    intended,
    distance,
    hand_force,
    region_force,
    distance_threshold,
    force_threshold,
    missing_weight,
):
    """Reward only intended hand contact with the configured object region."""

    if intended.shape != hand_force.shape or intended.ndim != 2:
        raise ValueError("intended and hand_force must share shape (envs, hands)")
    if distance.ndim != 3 or distance.shape[:2] != intended.shape:
        raise ValueError("distance must have shape (envs, hands, target_links)")
    if region_force.shape != (intended.shape[0], distance.shape[2]):
        raise ValueError("region_force must have shape (envs, target_links)")

    intended_contact = intended > 0.1
    near_region_by_link = distance <= distance_threshold
    loaded_hand = hand_force >= force_threshold
    loaded_region_by_link = region_force[:, None, :] >= force_threshold
    live_target_contact = loaded_hand & torch.any(
        near_region_by_link & loaded_region_by_link,
        dim=2,
    )
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
    proximity = torch.where(torch.isfinite(proximity), proximity, torch.zeros_like(proximity))
    score = torch.where(
        live_target_contact,
        torch.ones_like(proximity),
        0.5 * proximity,
    )
    hand_reward = torch.where(
        intended_contact,
        floor + (1.0 - floor) * score,
        torch.ones_like(score),
    )
    hand_error = (intended_contact & ~live_target_contact).to(dtype=distance.dtype)
    return hand_reward, hand_error, live_target_contact


def _opposing_contact_power_reward(
    *,
    intended,
    distance,
    hand_force,
    region_force,
    region_force_vector,
    reference_point_velocity,
    point_link_ids,
    distance_threshold,
    force_threshold,
    power_scale,
    positive_margin,
    error_scale,
    reward_floor,
):
    """Reward same-link contact power up to a small positive margin."""

    if intended.shape != hand_force.shape or intended.ndim != 2:
        raise ValueError("intended and hand_force must share shape (envs, hands)")
    if distance.ndim != 3 or distance.shape[:2] != intended.shape:
        raise ValueError("distance must have shape (envs, hands, target_links)")
    link_count = int(distance.shape[2])
    if region_force.shape != (intended.shape[0], link_count):
        raise ValueError("region_force must have shape (envs, target_links)")
    if region_force_vector.shape != (intended.shape[0], link_count, 3):
        raise ValueError(
            "region_force_vector must have shape (envs, target_links, 3)"
        )
    if (
        reference_point_velocity.ndim != 3
        or reference_point_velocity.shape[0] != intended.shape[0]
        or reference_point_velocity.shape[2] != 3
    ):
        raise ValueError(
            "reference_point_velocity must have shape (envs, points, 3)"
        )
    if point_link_ids.shape != (reference_point_velocity.shape[1],):
        raise ValueError("point_link_ids must contain one target-link id per point")
    if (
        power_scale <= 0.0
        or not torch.isfinite(torch.as_tensor(positive_margin)).item()
        or positive_margin < 0.0
        or error_scale < 0.0
    ):
        raise ValueError(
            "power_scale must be positive; positive_margin and error_scale "
            "must be finite and nonnegative"
        )
    if reward_floor < 0.0 or reward_floor > 1.0:
        raise ValueError("reward_floor must be in [0, 1]")

    same_link = (
        (intended > 0.1)[:, :, None]
        & (distance <= distance_threshold)
        & (hand_force >= force_threshold)[:, :, None]
        & (region_force >= force_threshold)[:, None, :]
    )
    active_link = torch.any(same_link, dim=1)
    force_by_point = region_force_vector[:, point_link_ids]
    point_power = torch.sum(
        force_by_point * reference_point_velocity,
        dim=-1,
    )
    conservative_link_power = []
    for link_index in range(link_count):
        link_points = point_power[:, point_link_ids == link_index]
        if link_points.shape[1] == 0:
            raise ValueError("Every target link must own at least one region point")
        conservative_link_power.append(link_points.amax(dim=1))
    conservative_link_power = torch.stack(conservative_link_power, dim=1)
    active_count = active_link.sum(dim=1)
    conservative_power = torch.sum(
        torch.where(
            active_link,
            conservative_link_power,
            torch.zeros_like(conservative_link_power),
        ),
        dim=1,
    ) / torch.clamp(active_count, min=1)
    power_error = torch.relu(
        (float(positive_margin) - conservative_power) / float(power_scale)
    ).pow(2)
    shaped_reward = float(reward_floor) + (
        1.0 - float(reward_floor)
    ) * torch.exp(-float(error_scale) * power_error)
    requires_power = torch.any(intended > 0.1, dim=1)
    reward = torch.where(
        requires_power,
        torch.where(
            active_count > 0,
            shaped_reward,
            torch.full_like(power_error, float(reward_floor)),
        ),
        torch.ones_like(power_error),
    )
    return reward, conservative_power, active_link


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
        from pipeline.physics.common_rollout import CommonRolloutRecorder, SMPLX_BODY_NAMES
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
        self._static_collision_filter = STATIC_SCENE_COLLISION_FILTER
        self._add_ground_plane = add_ground_plane
        self._validate_humanoid_object_collision_filters = (
            validate_humanoid_object_collision_filters
        )

        env = cfg["env"]
        manifest_path = Path(env["articulatedInputPath"]).expanduser().resolve()
        self._object_config = json.loads(manifest_path.read_text(encoding="utf-8"))
        object_reference_path = Path(
            self._object_config["reference_path"]
        ).expanduser().resolve()
        with np.load(object_reference_path, allow_pickle=False) as values:
            object_root_pos = np.asarray(values["object_root_pos"], dtype=np.float32)
            object_root_rot = np.asarray(values["object_root_rot_xyzw"], dtype=np.float32)
            self._q_reference_np = np.asarray(values["object_joint_qpos"], dtype=np.float32)
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
            self._reference_link_names = [
                str(value) for value in np.asarray(values["body_names"]).tolist()
            ]
            self._intended_contact_np = np.asarray(values["intended"], dtype=np.bool_)
            self._contact_points_np = np.asarray(
                values["contact_points_link_local_scaled"], dtype=np.float32
            )
            self._contact_point_link_names = [
                str(value)
                for value in np.asarray(values["contact_point_link_names"]).tolist()
            ]
            self._contact_region_link_names = [
                str(value)
                for value in np.asarray(values["contact_region_link_names"]).tolist()
            ]
            self._reference_fps = float(np.asarray(values["fps"]).item())
            parent_points = np.asarray(values["active_parent_points"], dtype=np.float32)
            child_points = np.asarray(values["active_child_points"], dtype=np.float32)
        self._object_creation_pos_np, self._object_creation_rot_np = _object_creation_pose(
            object_root_pos, object_root_rot
        )

        self._joint_names = [str(value) for value in self._object_config["joint_names"]]
        self._active_joint_names = [
            str(value) for value in self._object_config["active_joint_names"]
        ]
        self._active_dof_ids_np = np.asarray(
            [self._joint_names.index(name) for name in self._active_joint_names],
            dtype=np.int64,
        )
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
        self._active_dof_count = len(self._active_joint_names)
        self._active_link_names = [
            str(value)
            for value in self._object_config["active_child_link_names"]
        ]
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
        self._observation_variant = env["articulationObservation"]
        if self._observation_variant not in {
            "rigid_graph",
            "articulated_graph",
            "joint_state",
            "articulated_graph_joint_state",
        }:
            raise ValueError(
                f"Unknown articulation observation: {self._observation_variant}"
        )
        self._use_articulated_graph = (
            self._active_dof_count > 0
            and self._observation_variant
            in {"articulated_graph", "articulated_graph_joint_state"}
        )
        self._use_joint_state = (
            self._active_dof_count > 0
            and self._observation_variant
            in {"joint_state", "articulated_graph_joint_state"}
        )
        qvel_scale = np.asarray(
            env["articulationQvelScale"], dtype=np.float32
        ).reshape(-1)
        if qvel_scale.size == 1:
            qvel_scale = np.repeat(qvel_scale, self._active_dof_count)
        if qvel_scale.shape != (self._active_dof_count,):
            raise ValueError(
                "articulationQvelScale must be scalar or match active object DOFs, got "
                f"{qvel_scale.shape} for {self._active_dof_count} active DOFs"
            )
        if not np.all(np.isfinite(qvel_scale)) or np.any(qvel_scale <= 0):
            raise ValueError("articulationQvelScale must contain finite positive values")
        self._qvel_scale_np = qvel_scale
        self._contact_power_scale = float(env["articulationContactPowerScale"])
        self._contact_power_margin = float(
            env["articulationContactPowerMargin"]
        )
        if (
            not np.isfinite(self._contact_power_scale)
            or self._contact_power_scale <= 0.0
        ):
            raise ValueError("articulationContactPowerScale must be positive finite")
        if (
            not np.isfinite(self._contact_power_margin)
            or self._contact_power_margin < 0.0
        ):
            raise ValueError(
                "articulationContactPowerMargin must be nonnegative finite"
            )
        self._native_obs_size = int(env["numObs"])
        if self._use_joint_state:
            env["numObs"] = self._native_obs_size + 4 * self._active_dof_count
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
        if not np.all(np.isfinite(self._q_reference_np)):
            raise ValueError("Object q reference must be finite")
        if self._intended_contact_np.shape != (self._q_reference_np.shape[0], 2):
            raise ValueError("intended contact must have shape (frames, 2)")
        if self._contact_points_np.shape != (len(self._contact_point_link_names), 3):
            raise ValueError("Contact region points and point-link names disagree")
        if (
            not self._contact_region_link_names
            or len(set(self._contact_region_link_names))
            != len(self._contact_region_link_names)
            or set(self._contact_point_link_names)
            != set(self._contact_region_link_names)
        ):
            raise ValueError("Canonical contact-region names and points disagree")
        self._object_points_np = np.concatenate(
            (parent_points.reshape(-1, 3), child_points.reshape(-1, 3)),
            axis=0,
        )
        self._configure_articulated_actor = configure_articulated_actor
        self._create_static_box_actors = create_static_box_actors
        self._load_articulated_asset = load_articulated_asset
        self._load_static_box_assets = load_static_box_assets
        self._CommonRolloutRecorder = CommonRolloutRecorder
        self._common_human_body_names = SMPLX_BODY_NAMES

        self._q_reward_weight = env["articulationRewardWeight"]
        self._qvel_reward_weight = env["articulationQvelRewardWeight"]
        self._opposing_contact_power_reward_weight = env[
            "articulationOpposingContactPowerRewardWeight"
        ]
        self._link_reward_weight = env["articulationLinkRewardWeight"]
        self._q_reward_scale = env["articulationRewardScale"]
        self._qvel_reward_scale = env["articulationQvelRewardScale"]
        self._opposing_contact_power_reward_scale = env[
            "articulationOpposingContactPowerRewardScale"
        ]
        self._opposing_contact_power_reward_floor = env[
            "articulationOpposingContactPowerRewardFloor"
        ]
        if (
            self._opposing_contact_power_reward_floor < 0.0
            or self._opposing_contact_power_reward_floor > 1.0
        ):
            raise ValueError(
                "articulationOpposingContactPowerRewardFloor must be in [0, 1]"
            )
        self._link_reward_scale = env["articulationLinkRewardScale"]
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
        self._recorder = None
        self._rollout_next_frame = None
        self._rollout_written = False
        self._rollout_terminated = None
        self._contact_measurement_cache = None
        self._q_reference = None
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        self._q_reference = torch.as_tensor(self._q_reference_np, device=self.device)
        self._link_reference = torch.as_tensor(self._link_reference_np, device=self.device)
        self._link_reference_rot = torch.as_tensor(
            self._link_reference_rot_np, device=self.device
        )
        self._intended_contact = torch.as_tensor(self._intended_contact_np, device=self.device)
        self._active_dof_ids = torch.as_tensor(
            self._active_dof_ids_np,
            device=self.device,
            dtype=torch.long,
        )
        lower = np.asarray(self._target_dof_properties["lower"], dtype=np.float32)
        upper = np.asarray(self._target_dof_properties["upper"], dtype=np.float32)
        active_lower = lower[self._active_dof_ids_np]
        active_range = (upper - lower)[self._active_dof_ids_np]
        if self._active_dof_count and (
            not np.all(np.isfinite(active_range)) or np.any(active_range <= 0)
        ):
            raise ValueError("Active object DOFs require finite positive joint ranges")
        self._q_lower = torch.as_tensor(active_lower, device=self.device)
        self._q_range = torch.as_tensor(active_range, device=self.device)
        self._qvel_scale = torch.as_tensor(self._qvel_scale_np, device=self.device)
        if self._rollout_path:
            self._rollout_terminated = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.bool
            )
        self._resolve_rollout_body_indices()
        if self._rollout_path:
            if self.num_envs != 1:
                raise ValueError("common rollout recording requires exactly one environment")
            self._recorder = self._CommonRolloutRecorder(
                self._rollout_path,
                fps=self._rollout_fps,
                object_joint_qpos_reference=self._q_reference_np,
                joint_names=self._joint_names,
                joint_types=self._joint_types,
                intended=self._intended_contact_np,
                contact_region_link_names=self._target_contact_link_names,
            )

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
            self._object_points_np[None],
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
        self._create_static_box_actors(
            self.gym,
            env_ptr,
            env_id,
            self._static_box_assets,
            self._object_config,
            collision_filter=self._static_collision_filter,
        )

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
        super()._reset_target(env_ids)
        reset_qpos, reset_qvel = _initial_object_reset_state(
            to_torch(self._initial_qpos_np, device=self.device),
            self.progress_buf[env_ids],
        )
        self._target_dof_pos[env_ids] = reset_qpos
        self._target_dof_vel[env_ids] = reset_qvel
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
        if (
            self._recorder is None
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
        self._recorder.append(
            0,
            human_root_state=self._humanoid_root_states[0].detach().cpu().numpy(),
            human_dof_pos=self._dof_pos[0].detach().cpu().numpy(),
            human_body_state=body_state.detach().cpu().numpy(),
            object_root_state=self._target_states[0].detach().cpu().numpy(),
            object_joint_qpos=self._target_dof_pos[0].detach().cpu().numpy(),
            region_distance_m=distance.detach().cpu().numpy(),
            hand_force_n=np.zeros(2, dtype=np.float32),
            region_force_n=np.zeros(link_count, dtype=np.float32),
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
        if self._use_joint_state:
            native = torch.cat((native, self._joint_state_observation(env_ids)), dim=-1)
        self.obs_buf[env_ids] = native

    def _compute_observations_iter(self, hoi_data, env_ids=None, delta_t=1):
        if not self._use_articulated_graph:
            return super()._compute_observations_iter(hoi_data, env_ids, delta_t)
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
                self._compute_task_obs(env_ids, ref_obs),
            ),
            dim=-1,
        )
        ig_all, ig, ref_ig = self._articulated_graph_observation(
            env_ids, ref_obs, next_ts
        )
        return torch.cat((obs, ig_all, ref_ig - ig), dim=-1)

    def _joint_state_observation(self, env_ids):
        if self._active_dof_count == 0:
            return self._target_dof_pos[env_ids, :0]

        q = self._target_dof_pos[env_ids][:, self._active_dof_ids]
        qvel = self._target_dof_vel[env_ids][:, self._active_dof_ids]
        frame = self.progress_buf[env_ids]
        frame_1 = torch.clamp(frame + 1, max=self._q_reference.shape[0] - 1)
        frame_16 = torch.clamp(frame + 16, max=self._q_reference.shape[0] - 1)
        normalized_q = 2.0 * (q - self._q_lower) / self._q_range - 1.0
        return torch.cat(
            (
                normalized_q,
                qvel / self._qvel_scale,
                (
                    self._q_reference[frame_1][:, self._active_dof_ids] - q
                ) / self._q_range,
                (
                    self._q_reference[frame_16][:, self._active_dof_ids] - q
                ) / self._q_range,
            ),
            dim=-1,
        )

    def _articulated_graph_observation(self, env_ids, ref_obs, next_ts):
        live_state = self._target_body_state[env_ids][
            :, self._contact_point_body_ids
        ]
        local_points = self._contact_points.unsqueeze(0).expand(
            len(env_ids), -1, -1
        )
        live_points = torch_utils.quat_rotate(
            live_state[..., 3:7].reshape(-1, 4), local_points.reshape(-1, 3)
        ).view(len(env_ids), -1, 3) + live_state[..., :3]

        ref_pos = self._link_reference[next_ts][
            :, self._graph_reference_link_ids
        ]
        ref_rot = self._link_reference_rot[next_ts][
            :, self._graph_reference_link_ids
        ]
        ref_points = torch_utils.quat_rotate(
            ref_rot.reshape(-1, 4), local_points.reshape(-1, 3)
        ).view(len(env_ids), -1, 3) + ref_pos

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

    def _compute_reset(self):
        super()._compute_reset()
        if not self._rollout_path:
            return

        self._rollout_terminated |= self._terminate_buf.bool()
        self._terminate_buf[:] = self._rollout_terminated.to(self._terminate_buf.dtype)

    def compute_obj_reward(self, weights):
        self._contact_measurement_cache = None
        native, reset, obj_points, ref_obj_points = super().compute_obj_reward(weights)
        frames = self._reference_frame()
        if self._active_dof_count:
            q = self._target_dof_pos[:, self._active_dof_ids]
            q_reference = self._q_reference[frames][:, self._active_dof_ids]
            q_error = torch.mean(
                ((q - q_reference) / self._q_range) ** 2,
                dim=1,
            )
            q_reward = torch.exp(-self._q_reward_scale * q_error)
            next_frames = torch.clamp(
                frames + 1, max=self._q_reference.shape[0] - 1
            )
            reference_qvel = (
                self._q_reference[next_frames][:, self._active_dof_ids]
                - q_reference
            ) * self._rollout_fps
            qvel_error = torch.mean(
                (
                    (
                        self._target_dof_vel[:, self._active_dof_ids]
                        - reference_qvel
                    )
                    / self._qvel_scale
                ) ** 2,
                dim=1,
            )
            qvel_reward = torch.exp(-self._qvel_reward_scale * qvel_error)
        else:
            q_reward = torch.ones(self.num_envs, device=self.device)
            qvel_reward = torch.ones_like(q_reward)
        opposing_contact_power_reward = torch.ones_like(q_reward)
        conservative_contact_power = torch.zeros_like(q_reward)
        if self._opposing_contact_power_reward_weight > 0.0:
            intended, distance, hand_force, region_force = (
                self._measure_reference_contacts(frames)
            )
            (
                opposing_contact_power_reward,
                conservative_contact_power,
                _,
            ) = _opposing_contact_power_reward(
                intended=intended,
                distance=distance,
                hand_force=hand_force,
                region_force=region_force,
                region_force_vector=self._target_contact_forces[
                    :, self._target_contact_body_ids
                ],
                reference_point_velocity=(
                    self._reference_contact_point_velocity(frames)
                ),
                point_link_ids=self._contact_point_link_ids,
                distance_threshold=self._target_contact_distance_threshold,
                force_threshold=self._target_contact_force_threshold,
                power_scale=self._contact_power_scale,
                positive_margin=self._contact_power_margin,
                error_scale=self._opposing_contact_power_reward_scale,
                reward_floor=self._opposing_contact_power_reward_floor,
            )
        link_reward = torch.ones_like(q_reward)
        if self._live_link_ids.numel() > 0:
            live = self._target_body_state[:, self._live_link_ids, :3]
            ref = self._link_reference[frames][:, self._reference_link_ids]
            link_reward = torch.exp(
                -self._link_reward_scale * torch.mean((live - ref) ** 2, dim=(1, 2))
            )
        scale = (
            torch.pow(q_reward, self._q_reward_weight)
            * torch.pow(qvel_reward, self._qvel_reward_weight)
            * torch.pow(
                opposing_contact_power_reward,
                self._opposing_contact_power_reward_weight,
            )
            * torch.pow(link_reward, self._link_reward_weight)
        )
        self.extras["articulation_q_reward"] = q_reward
        self.extras["articulation_qvel_reward"] = qvel_reward
        self.extras[
            "articulation_opposing_contact_power_reward"
        ] = opposing_contact_power_reward
        self.extras[
            "articulation_conservative_contact_power_w"
        ] = conservative_contact_power
        self.extras["articulation_link_reward"] = link_reward
        return native * scale, reset, obj_points, ref_obj_points

    def compute_cg_reward(self, weights):
        """Match hand2 labels to contact with the configured target region."""

        contact_threshold = 0.1
        frames = self._reference_frame()
        cached = getattr(self, "_contact_measurement_cache", None)
        if cached is not None and torch.equal(cached[0], frames):
            _, intended, distance, hand_force, region_force = cached
        else:
            intended = (
                self._intended_contact[frames] > contact_threshold
            ).float()
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
                torch.any(intended > contact_threshold, dim=1),
                as_tuple=False,
            ).reshape(-1)
            if active_env_ids.numel() > 0:
                active_distance, active_hand_force, active_region_force = (
                    self._measure_contacts(active_env_ids)
                )
                distance[active_env_ids] = active_distance
                hand_force[active_env_ids] = active_hand_force
                region_force[active_env_ids] = active_region_force

        hand_reward, hand_error, live_target_contact = (
            _target_region_contact_reward(
                intended=intended,
                distance=distance,
                hand_force=hand_force,
                region_force=region_force,
                distance_threshold=self._target_contact_distance_threshold,
                force_threshold=self._target_contact_force_threshold,
                missing_weight=weights["cg_hand"],
            )
        )
        human_contact = self.extract_data_component(
            "contact_human", obs=self._curr_obs
        )

        # Keep the native non-hand contact and total-contact energy terms. Hand
        # references are deliberately absent here because their only GT is hand2.
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
        self.extras["target_contact_reward"] = hand_reward.mean(dim=1)
        self.extras["target_contact_live"] = live_target_contact.float().mean(dim=1)
        nearest_distance = distance.amin(dim=2)
        self.extras["target_contact_distance_m"] = torch.where(
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
        self._live_link_ids = torch.as_tensor(
            [body_lookup[name] for name in self._active_link_names],
            device=self.device,
            dtype=torch.long,
        )
        self._reference_link_ids = torch.as_tensor(
            [reference_lookup[name] for name in self._active_link_names],
            device=self.device,
            dtype=torch.long,
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
            self._contact_region_link_names
        )
        if not self._target_contact_link_names:
            raise ValueError("Contact region must contain at least one object link")
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
        if self._use_articulated_graph:
            if not self._contact_point_link_names:
                raise ValueError("articulated_graph requires contact-region points")
            self._graph_reference_link_ids = (
                self._contact_point_reference_link_ids
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

    def _reference_contact_point_velocity(self, frames):
        next_frames = torch.clamp(
            frames + 1,
            max=self._link_reference.shape[0] - 1,
        )

        def world_points(frame_ids):
            pos = self._link_reference[frame_ids][
                :, self._contact_point_reference_link_ids
            ]
            rot = self._link_reference_rot[frame_ids][
                :, self._contact_point_reference_link_ids
            ]
            local = self._contact_points.unsqueeze(0).expand(
                frame_ids.shape[0], -1, -1
            )
            return pos + torch_utils.quat_rotate(
                rot.reshape(-1, 4),
                local.reshape(-1, 3),
            ).view_as(local)

        return (
            world_points(next_frames) - world_points(frames)
        ) * self._rollout_fps

    def post_physics_step(self):
        super().post_physics_step()
        if self._recorder is None or self._rollout_written:
            return
        distance, hand_force, region_force = self._measure_contacts()
        frame = int(self._reference_frame()[0].item())
        if self._rollout_next_frame is None:
            return
        if frame != self._rollout_next_frame:
            raise RuntimeError(
                "InterMimic recorder expected reference frame "
                f"{self._rollout_next_frame}, got {frame}"
            )
        self._recorder.append(
            frame,
            human_root_state=self._humanoid_root_states[0].detach().cpu().numpy(),
            human_dof_pos=self._dof_pos[0].detach().cpu().numpy(),
            human_body_state=self._rigid_body_state.view(
                self.num_envs, -1, 13
            )[0, :self.num_bodies].detach().cpu().numpy(),
            object_root_state=self._target_states[0].detach().cpu().numpy(),
            object_joint_qpos=self._target_dof_pos[0].detach().cpu().numpy(),
            region_distance_m=distance[0].detach().cpu().numpy(),
            hand_force_n=hand_force[0].detach().cpu().numpy(),
            region_force_n=region_force[0].detach().cpu().numpy(),
        )
        self._rollout_next_frame += 1
        done = bool(self.reset_buf[0].item()) or frame >= self._q_reference.shape[0] - 1
        if done:
            self._recorder.seal()
            self._rollout_written = True


__all__ = ["InterMimicArticulated"]
