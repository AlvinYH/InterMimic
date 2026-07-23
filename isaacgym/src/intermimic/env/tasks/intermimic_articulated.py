"""InterMimic with the smallest articulated-object task extension."""

from __future__ import annotations

import json
from pathlib import Path

from isaacgym import gymapi, gymtorch
import numpy as np
import torch
import trimesh

from .intermimic import InterMimic, compute_sdf
from isaacgym.torch_utils import to_torch
from ...utils import torch_utils
from pipeline.physics.mimic.collision_distance import (
    box_region_surface_distances,
    capsule_region_surface_distances,
    load_mjcf_body_boxes,
    load_mjcf_body_capsules,
)
from pipeline.physics.mimic.contact_pairs import (
    aggregate_rigid_contact_groups,
    hand2_body_groups,
)


_ARTICULATED_OBJECT_COLLISION_FILTER = 2


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


def _creation_dof_state(state, initial_qpos):
    if state.shape != (len(initial_qpos),):
        raise ValueError(
            f"Actor DOF state and JSON q0 disagree: {state.shape} vs {initial_qpos.shape}"
        )
    if state.dtype.names is None or not {"pos", "vel"}.issubset(state.dtype.names):
        raise ValueError("Actor DOF state must expose pos/vel fields")
    state = state.copy()
    state["pos"] = initial_qpos
    state["vel"] = 0.0
    return state


class InterMimicArticulated(InterMimic):
    """Keep native InterMimic PPO and add passive object q/link task state."""

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        env = cfg["env"]
        self._object_config_path = Path(env["articulatedObjectConfigPath"]).expanduser().resolve()
        self._object_config = json.loads(self._object_config_path.read_text(encoding="utf-8"))
        object_reference_path = Path(env["articulatedObjectReferencePath"]).expanduser().resolve()
        with np.load(object_reference_path, allow_pickle=False) as values:
            object_root_pos = np.asarray(values["object_root_pos"], dtype=np.float32)
            object_root_rot = np.asarray(values["object_root_rot_xyzw"], dtype=np.float32)
            self._q_reference_np = np.asarray(values["object_joint_qpos"], dtype=np.float32)
            self._link_reference_np = np.asarray(values["object_link_pos"], dtype=np.float32)
            self._link_reference_rot_np = np.asarray(
                values["object_link_rot_xyzw"], dtype=np.float32
            )
            self._reference_link_names = [str(v) for v in np.asarray(values["link_names"]).tolist()]
        self._object_creation_pos_np, self._object_creation_rot_np = _object_creation_pose(
            object_root_pos, object_root_rot
        )

        self._joint_names = [str(v) for v in self._object_config["articulated_target_joint_names"]]
        self._initial_qpos_np = np.asarray(self._object_config["initial_joint_qpos"], dtype=np.float32)
        self._object_dof_count = len(self._joint_names)
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
            self._object_dof_count > 0
            and self._observation_variant
            in {"articulated_graph", "articulated_graph_joint_state"}
        )
        self._use_joint_state = (
            self._object_dof_count > 0
            and self._observation_variant
            in {"joint_state", "articulated_graph_joint_state"}
        )
        qvel_scale = np.asarray(
            env["articulationQvelScale"], dtype=np.float32
        ).reshape(-1)
        if qvel_scale.size == 1:
            qvel_scale = np.repeat(qvel_scale, self._object_dof_count)
        if qvel_scale.shape != (self._object_dof_count,):
            raise ValueError(
                "articulationQvelScale must be scalar or match object DOFs, got "
                f"{qvel_scale.shape} for {self._object_dof_count} DOFs"
            )
        if not np.all(np.isfinite(qvel_scale)) or np.any(qvel_scale <= 0):
            raise ValueError("articulationQvelScale must contain finite positive values")
        self._qvel_scale_np = qvel_scale
        self._native_obs_size = int(env["numObs"])
        if self._use_joint_state:
            env["numObs"] = self._native_obs_size + 4 * self._object_dof_count
        if (
            self._q_reference_np.ndim != 2
            or self._q_reference_np.shape[1] != self._object_dof_count
        ):
            raise ValueError(
                "object_joint_qpos must have shape (frames, object DOFs), got "
                f"{self._q_reference_np.shape} for {self._object_dof_count} DOFs"
            )
        if self._initial_qpos_np.shape != (self._object_dof_count,):
            raise ValueError(
                "initial_joint_qpos must match articulated_target_joint_names, got "
                f"{self._initial_qpos_np.shape} for {self._object_dof_count} joints"
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
        if not np.all(np.isfinite(self._q_reference_np)) or not np.all(
            np.isfinite(self._initial_qpos_np)
        ):
            raise ValueError("Object q reference and initial_joint_qpos must be finite")

        self._q_reward_weight = env["articulationRewardWeight"]
        self._link_reward_weight = env["articulationLinkRewardWeight"]
        self._q_reward_scale = env["articulationRewardScale"]
        self._link_reward_scale = env["articulationLinkRewardScale"]
        self._rollout_path = env["rolloutOutputPath"]
        self._rollout_fps = float(env["dataFPS"])
        if self._rollout_fps <= 0.0:
            raise ValueError("dataFPS must be positive")
        if self._rollout_path and bool(sim_params.use_gpu_pipeline):
            raise RuntimeError(
                "Exact articulated rollout contact requires --pipeline cpu"
            )
        self._humanoid_mjcf_path = Path(
            env["articulatedHumanoidXmlPath"]
        ).expanduser().resolve()

        contact_reference_path = Path(env["contactReferencePath"]).expanduser().resolve()
        with np.load(contact_reference_path, allow_pickle=False) as values:
            self._intended_contact_np = np.asarray(values["contact_labels"], dtype=np.float32)
            self._contact_label_names = [
                str(v) for v in np.asarray(values["contact_label_names"]).tolist()
            ]
            self._contact_granularity = str(
                np.asarray(values["contact_granularity"]).reshape(())
            )
        if self._intended_contact_np.shape != (
            self._q_reference_np.shape[0],
            len(self._contact_label_names),
        ):
            raise ValueError(
                "contact_labels must have shape (frames, labels), got "
                f"{self._intended_contact_np.shape}"
            )
        if self._contact_granularity != "hand2" or self._contact_label_names != [
            "left_hand",
            "right_hand",
        ]:
            raise ValueError(
                "Canonical contact reference must be hand2 ordered as left_hand/right_hand"
            )

        contact_points_path = Path(
            self._object_config["contact_region_points_path"]
        ).expanduser().resolve()
        with np.load(contact_points_path, allow_pickle=False) as values:
            self._contact_points_np = np.asarray(
                values["points_link_local_scaled"], dtype=np.float32
            )
            self._contact_point_link_names = [
                str(v) for v in np.asarray(values["point_link_names"]).tolist()
            ]
        if self._contact_points_np.shape != (len(self._contact_point_link_names), 3):
            raise ValueError(
                "Contact region points and point_link_names disagree: "
                f"{self._contact_points_np.shape} vs {len(self._contact_point_link_names)} names"
            )

        self._rollout = []
        self._rollout_written = False
        self._rollout_terminated = None
        self._last_env0_reset_qpos = None
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        self._q_reference = torch.as_tensor(self._q_reference_np, device=self.device)
        self._link_reference = torch.as_tensor(self._link_reference_np, device=self.device)
        self._link_reference_rot = torch.as_tensor(
            self._link_reference_rot_np, device=self.device
        )
        self._intended_contact = torch.as_tensor(self._intended_contact_np, device=self.device)
        lower = np.asarray(self._target_dof_properties["lower"], dtype=np.float32)
        upper = np.asarray(self._target_dof_properties["upper"], dtype=np.float32)
        joint_range = upper - lower
        if self._object_dof_count and (
            not np.all(np.isfinite(joint_range)) or np.any(joint_range <= 0)
        ):
            raise ValueError("Articulated object DOFs require finite positive joint ranges")
        self._q_lower = torch.as_tensor(lower, device=self.device)
        self._q_range = torch.as_tensor(joint_range, device=self.device)
        self._qvel_scale = torch.as_tensor(self._qvel_scale_np, device=self.device)
        if self._rollout_path:
            self._rollout_terminated = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.bool
            )
        self._resolve_rollout_body_indices()

    def _load_target_asset(self):
        urdf_path = Path(self._object_config["urdf_path"]).expanduser().resolve()
        options = gymapi.AssetOptions()
        options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        options.vhacd_enabled = self._object_config["isaac_vhacd_enabled"]
        asset = self.gym.load_asset(self.sim, str(urdf_path.parent), urdf_path.name, options)
        asset_joint_names = list(self.gym.get_asset_dof_names(asset))
        if asset_joint_names != self._joint_names:
            raise ValueError(
                "Loaded URDF DOF names do not match articulated_target_joint_names: "
                f"{asset_joint_names} vs {self._joint_names}"
            )
        self._target_asset = [asset]
        self._target_dof_properties = self.gym.get_asset_dof_properties(asset)
        self._target_dof_properties["driveMode"].fill(gymapi.DOF_MODE_NONE)
        self._target_dof_properties["stiffness"].fill(0.0)
        self._target_dof_properties["damping"].fill(
            self._object_config["object_joint_damping"]
        )
        self._target_dof_properties["friction"].fill(
            self._object_config["object_joint_friction"]
        )
        self._target_asset_body_names = list(self.gym.get_asset_rigid_body_names(asset))
        self.target_aggregate_body_capacity = self.gym.get_asset_rigid_body_count(asset)
        self.target_aggregate_shape_capacity = self.gym.get_asset_rigid_shape_count(asset)

        mesh_path = Path(self._object_config["object_mesh_path"]).expanduser().resolve()
        mesh = trimesh.load(mesh_path, force="mesh")
        points, _ = trimesh.sample.sample_surface(mesh, 1024, seed=2024)
        scale = np.asarray(self._object_config["object_scale"], dtype=np.float32).reshape(-1)
        if scale.size == 1:
            scale = np.repeat(scale, 3)
        points = points.astype(np.float32) * scale[None]
        self.object_points = torch.as_tensor(points[None], device=self.device)

    def _build_target(self, env_id, env_ptr):
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*self._object_creation_pos_np)
        pose.r = gymapi.Quat(*self._object_creation_rot_np)
        handle = self.gym.create_actor(
            env_ptr,
            self._target_asset[0],
            pose,
            self._object_config["object_name"],
            env_id,
            _ARTICULATED_OBJECT_COLLISION_FILTER,
            0,
        )
        self.gym.set_actor_dof_properties(env_ptr, handle, self._target_dof_properties)
        creation_state = self.gym.get_actor_dof_states(
            env_ptr, handle, gymapi.STATE_ALL
        )
        creation_state = _creation_dof_state(creation_state, self._initial_qpos_np)
        self.gym.set_actor_dof_states(
            env_ptr, handle, creation_state, gymapi.STATE_ALL
        )
        self._target_handles.append(handle)

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
        self._target_dof_pos[env_ids] = to_torch(self._initial_qpos_np, device=self.device)
        self._target_dof_vel[env_ids] = 0.0
        if self._rollout_terminated is not None:
            self._rollout_terminated[env_ids] = False
        if torch.any(env_ids == 0):
            self._last_env0_reset_qpos = self._target_dof_pos[0].detach().cpu().numpy().copy()

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        actor_ids = self._tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_states), gymtorch.unwrap_tensor(actor_ids), len(actor_ids)
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._dof_state), gymtorch.unwrap_tensor(actor_ids), len(actor_ids)
        )

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        if self._pd_control:
            human = self._action_to_pd_targets(self.actions)
            targets = torch.cat((human, self._target_dof_pos), dim=1).contiguous()
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))
        else:
            human = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
            forces = torch.cat((human, torch.zeros_like(self._target_dof_pos)), dim=1).contiguous()
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(forces))

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
        if self._object_dof_count == 0:
            return self._target_dof_pos[env_ids]

        q = self._target_dof_pos[env_ids]
        qvel = self._target_dof_vel[env_ids]
        frame = self.progress_buf[env_ids]
        frame_1 = torch.clamp(frame + 1, max=self._q_reference.shape[0] - 1)
        frame_16 = torch.clamp(frame + 16, max=self._q_reference.shape[0] - 1)
        normalized_q = 2.0 * (q - self._q_lower) / self._q_range - 1.0
        return torch.cat(
            (
                normalized_q,
                qvel / self._qvel_scale,
                (self._q_reference[frame_1] - q) / self._q_range,
                (self._q_reference[frame_16] - q) / self._q_range,
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
        native, reset, obj_points, ref_obj_points = super().compute_obj_reward(weights)
        frames = self._reference_frame()
        if self._object_dof_count:
            q_error = torch.mean(
                (
                    (self._target_dof_pos - self._q_reference[frames])
                    / self._q_range
                ) ** 2,
                dim=1,
            )
            q_reward = torch.exp(-self._q_reward_scale * q_error)
        else:
            q_reward = torch.ones(self.num_envs, device=self.device)
        link_reward = torch.ones_like(q_reward)
        if self._live_link_ids.numel() > 0:
            live = self._target_body_state[:, self._live_link_ids, :3]
            ref = self._link_reference[frames][:, self._reference_link_ids]
            link_reward = torch.exp(
                -self._link_reward_scale * torch.mean((live - ref) ** 2, dim=(1, 2))
            )
        scale = torch.pow(q_reward, self._q_reward_weight) * torch.pow(
            link_reward, self._link_reward_weight
        )
        self.extras["articulation_q_reward"] = q_reward
        self.extras["articulation_link_reward"] = link_reward
        return native * scale, reset, obj_points, ref_obj_points

    def compute_cg_reward(self, weights):
        """Use the hand2 sidecar without inventing per-finger reference labels."""

        contact_threshold = 0.1
        frames = self._reference_frame()
        intended = (self._intended_contact[frames] > contact_threshold).float()
        human_contact = self.extract_data_component(
            "contact_human", obs=self._curr_obs
        )
        live = torch.stack(
            [
                torch.any(
                    human_contact[:, body_ids] > contact_threshold, dim=1
                ).float()
                for body_ids in self._human_contact_body_groups
            ],
            dim=1,
        )

        # Native InterMimic only gates required hand contact. A side is therefore
        # one contact target, independent of how many finger bodies touch it.
        hand_error = intended * torch.abs(live - intended)
        hand_reward = 0.5 * (
            1.0 + torch.exp(-hand_error * weights["cg_hand"])
        )

        # Keep the native non-hand contact and total-contact energy terms. Hand
        # references are deliberately absent here because their only GT is hand2.
        hand_body_ids = {
            body_id
            for body_group in self._human_contact_body_groups
            for body_id in body_group
        }
        other_ids = [
            body_id
            for body_id in range(len(self.contact_bodies))
            if body_id not in hand_body_ids
        ]
        ref_other_contact = self.extract_data_component(
            "contact_human", obs=self._curr_ref_obs
        )[:, other_ids]
        other_contact = human_contact[:, other_ids]
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
        return reward, hand_error

    def _resolve_rollout_body_indices(self):
        body_lookup = {name: i for i, name in enumerate(self._target_asset_body_names)}
        missing_links = [name for name in self._reference_link_names if name not in body_lookup]
        if missing_links:
            raise ValueError(f"Object reference links are absent from the loaded URDF: {missing_links}")
        self._live_link_ids = torch.as_tensor(
            [body_lookup[name] for name in self._reference_link_names],
            device=self.device,
            dtype=torch.long,
        )
        self._reference_link_ids = torch.arange(
            len(self._reference_link_names), device=self.device, dtype=torch.long
        )

        human_names = list(self.gym.get_actor_rigid_body_names(self.envs[0], self.humanoid_handles[0]))
        self._human_contact_body_groups = hand2_body_groups(human_names)
        flat_body_ids = tuple(
            body_id for group in self._human_contact_body_groups for body_id in group
        )
        capsules = load_mjcf_body_capsules(
            self._humanoid_mjcf_path,
            [human_names[body_id] for body_id in flat_body_ids],
        )
        boxes = load_mjcf_body_boxes(
            self._humanoid_mjcf_path,
            [human_names[body_id] for body_id in flat_body_ids],
        )
        if not bool(np.logical_or(capsules.valid, boxes.valid).all()):
            missing = [
                name
                for name, capsule_valid, box_valid in zip(
                    capsules.body_names,
                    capsules.valid.tolist(),
                    boxes.valid.tolist(),
                )
                if not capsule_valid and not box_valid
            ]
            raise ValueError(f"Hand collision bodies have no capsule or box geometry: {missing}")
        self._human_contact_body_ids = torch.as_tensor(
            flat_body_ids, device=self.device, dtype=torch.long
        )
        self._human_contact_group_slices = (
            slice(0, len(self._human_contact_body_groups[0])),
            slice(len(self._human_contact_body_groups[0]), len(flat_body_ids)),
        )
        self._human_contact_capsule_endpoints = torch.as_tensor(
            capsules.endpoints_local, device=self.device
        )
        self._human_contact_capsule_radii = torch.as_tensor(
            capsules.radii, device=self.device
        )
        self._human_contact_capsule_valid = torch.as_tensor(
            capsules.valid, device=self.device
        )
        self._human_contact_box_centers = torch.as_tensor(
            boxes.centers_local, device=self.device
        )
        self._human_contact_box_quaternions = torch.as_tensor(
            boxes.quaternions_local_xyzw, device=self.device
        )
        self._human_contact_box_half_extents = torch.as_tensor(
            boxes.half_extents, device=self.device
        )
        self._human_contact_box_valid = torch.as_tensor(
            boxes.valid, device=self.device
        )

        region_names = set(self._contact_point_link_names)
        missing_region_links = sorted(region_names.difference(body_lookup))
        if missing_region_links:
            raise ValueError(f"Contact region links are absent from the loaded URDF: {missing_region_links}")
        target_local = [
            i for i, name in enumerate(self._target_asset_body_names)
            if not region_names or name in region_names
        ]
        env = self.envs[0]
        self._human_contact_env_body_groups = tuple(
            tuple(
                int(
                    self.gym.get_actor_rigid_body_index(
                        env,
                        self.humanoid_handles[0],
                        body_id,
                        gymapi.DOMAIN_ENV,
                    )
                )
                for body_id in group
            )
            for group in self._human_contact_body_groups
        )
        self._target_contact_env_body_ids = [
            int(
                self.gym.get_actor_rigid_body_index(
                    env,
                    self._target_handles[0],
                    body_id,
                    gymapi.DOMAIN_ENV,
                )
            )
            for body_id in target_local
        ]

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
        if self._use_articulated_graph:
            if not self._contact_point_link_names:
                raise ValueError("articulated_graph requires contact-region points")
            reference_lookup = {
                name: i for i, name in enumerate(self._reference_link_names)
            }
            missing_reference_links = sorted(
                set(self._contact_point_link_names).difference(reference_lookup)
            )
            if missing_reference_links:
                raise ValueError(
                    "Contact-region links are absent from object reference: "
                    f"{missing_reference_links}"
                )
            self._graph_reference_link_ids = torch.as_tensor(
                [reference_lookup[name] for name in self._contact_point_link_names],
                device=self.device,
                dtype=torch.long,
            )

    def _measure_contacts(self):
        distance = torch.full((self.num_envs, 2), float("inf"), device=self.device)
        if self._contact_points.numel() > 0:
            state = self._target_body_state[:, self._contact_point_body_ids]
            points = torch_utils.quat_rotate(
                state[..., 3:7].reshape(-1, 4),
                self._contact_points.unsqueeze(0).expand(self.num_envs, -1, -1).reshape(-1, 3),
            ).view(self.num_envs, -1, 3) + state[..., :3]
            human_state = self._rigid_body_state.view(self.num_envs, -1, 13)[
                :, self._human_contact_body_ids
            ]
            body_distance = capsule_region_surface_distances(
                body_pos=human_state[..., :3],
                body_quat_xyzw=human_state[..., 3:7],
                endpoints_local=self._human_contact_capsule_endpoints,
                radii=self._human_contact_capsule_radii,
                valid=self._human_contact_capsule_valid,
                region_points=points,
            )
            box_distance = box_region_surface_distances(
                body_pos=human_state[..., :3],
                body_quat_xyzw=human_state[..., 3:7],
                centers_local=self._human_contact_box_centers,
                quaternions_local_xyzw=self._human_contact_box_quaternions,
                half_extents=self._human_contact_box_half_extents,
                valid=self._human_contact_box_valid,
                region_points=points,
            )
            body_distance = torch.minimum(body_distance, box_distance)
            for label, group in enumerate(self._human_contact_group_slices):
                distance[:, label] = body_distance[:, group].min(dim=1).values

        exact = aggregate_rigid_contact_groups(
            self.gym.get_env_rigid_contacts(self.envs[0]),
            humanoid_body_groups=self._human_contact_env_body_groups,
            target_body_indices=self._target_contact_env_body_ids,
        )
        active = exact.count.sum(axis=1, dtype=np.int32) > 0
        return distance, active[None, :]

    def post_physics_step(self):
        super().post_physics_step()
        if not self._rollout_path or self._rollout_written or self.num_envs != 1:
            return
        distance, active = self._measure_contacts()
        frame = int(self._reference_frame()[0].item())
        self._rollout.append({
            "human_root_state": self._humanoid_root_states[0].detach().cpu().numpy(),
            "human_dof_pos": self._dof_pos[0].detach().cpu().numpy(),
            "human_body_state": self._rigid_body_state.view(self.num_envs, -1, 13)[0, :self.num_bodies].detach().cpu().numpy(),
            "object_root_state": self._target_states[0].detach().cpu().numpy(),
            "object_joint_qpos": self._target_dof_pos[0].detach().cpu().numpy(),
            "object_joint_qpos_reference": self._q_reference[frame].detach().cpu().numpy(),
            "object_link_state": self._target_body_state[0].detach().cpu().numpy(),
            "region_distance_m": distance[0].detach().cpu().numpy(),
            "intended": self._intended_contact[frame].detach().cpu().numpy(),
            "active": active[0],
            "terminated": bool(self._terminate_buf[0].item()),
        })
        done = bool(self.reset_buf[0].item()) or frame >= self._q_reference.shape[0] - 1
        if done:
            self._write_articulated_rollout()

    def _write_articulated_rollout(self):
        if not self._rollout:
            return
        if self._last_env0_reset_qpos is None:
            raise RuntimeError("Rollout finished before environment 0 recorded its object reset qpos")
        path = Path(self._rollout_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = self._rollout[0].keys()
        payload = {key: np.stack([frame[key] for frame in self._rollout]) for key in keys}
        valid_frames = len(self._rollout)
        total_frames = self._q_reference.shape[0] - 1
        if valid_frames > total_frames:
            raise ValueError(
                f"InterMimic rollout has {valid_frames} frames for a {total_frames}-frame reference"
            )
        missing_frames = total_frames - valid_frames
        if missing_frames:
            for key, values in payload.items():
                if key in {"object_joint_qpos_reference", "intended"}:
                    continue
                if key == "terminated":
                    padding = np.ones((missing_frames,), dtype=values.dtype)
                elif key == "active":
                    padding = np.zeros((missing_frames,) + values.shape[1:], dtype=values.dtype)
                else:
                    padding = np.repeat(values[-1:], missing_frames, axis=0)
                payload[key] = np.concatenate((values, padding), axis=0)
        payload["object_joint_qpos_reference"] = self._q_reference[1:].detach().cpu().numpy()
        payload["intended"] = (
            self._intended_contact[1:].detach().cpu().numpy() > 0.5
        )
        payload.update({
            "fps": np.asarray(self._rollout_fps, dtype=np.float32),
            "frame_id": np.arange(1, total_frames + 1, dtype=np.int64),
            "joint_names": np.asarray(self._joint_names),
            "link_names": np.asarray(self._target_asset_body_names),
            "contact_label_names": np.asarray(self._contact_label_names),
            "contact_granularity": np.asarray(self._contact_granularity),
            "contact_semantics": np.asarray("exact_rigid_pair"),
            "q0_source": np.asarray("case_json.object.initial_joint_values"),
            "reset_object_joint_qpos": self._last_env0_reset_qpos,
            "valid_frame_count": np.asarray(valid_frames, dtype=np.int64),
        })
        np.savez_compressed(path, **payload)
        self._rollout_written = True


__all__ = ["InterMimicArticulated"]
