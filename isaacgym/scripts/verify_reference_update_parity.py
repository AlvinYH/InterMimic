#!/usr/bin/env python3
"""GPU parity check for the tensorized InterMimic reference update."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

from isaacgym import gymapi
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intermimic.env.tasks.intermimic import InterMimic, compute_sdf


def _old_update(task) -> None:
    reset_ind = task.reset_buf == 1
    data_id = task.data_id[reset_ind]
    max_episode_length = task.max_episode_length[data_id]
    if (max_episode_length < task.rollout_length).all():
        task._sum_reward[reset_ind] = 0
        return
    start_index = task.start_times[reset_ind]
    end_index = task.progress_buf[reset_ind]
    task._sum_reward[reset_ind].mean()
    if torch.rand(1)[0] < 0:
        task._sum_reward[reset_ind] = 0
        return
    task._sum_reward[reset_ind] = 0
    reset_ind = torch.logical_and(
        reset_ind,
        task.max_episode_length[task.data_id] > task.rollout_length,
    )
    if reset_ind.sum() < 0.995:
        return
    curr_reward = task._curr_reward[reset_ind]
    state = task._curr_state[reset_ind]
    reward = torch.zeros(
        (curr_reward.shape[0], task.hoi_refs.shape[0], task.hoi_refs.shape[2]),
        device=curr_reward.device,
    )
    end_i = torch.minimum(
        max_episode_length,
        task.rollout_length + start_index,
    )
    assert (end_index < end_i).all()
    for row in range(curr_reward.shape[0]):
        if end_index[row] > start_index[row] + 30:
            frame = torch.arange(
                start_index[row] + 10,
                end_index[row] - 10,
                device=start_index.device,
            )
            reward[
                row,
                data_id[row],
                start_index[row] + 10:end_index[row] - 10,
            ] = (
                (end_index[row] - frame)
                / (end_i[row] - frame)
            )
    adjust_reward, winner = reward.max(dim=0)
    for motion in range(reward.shape[1]):
        if task.max_episode_length[motion] < task.rollout_length:
            continue
        for frame in range(reward.shape[2]):
            if task.max_episode_length[motion] - frame < task.rollout_length:
                break
            _, slot = task.ref_reward[motion, 1:, frame].min(dim=0)
            slot = slot + 1
            source = winner[motion, frame]
            source_frame = frame - start_index[source]
            if (
                source_frame > 0
                and source_frame < task.rollout_length
                and adjust_reward[motion, frame] > 0.5
            ):
                task.ref_reward[motion, slot, frame] = adjust_reward[
                    motion, frame
                ]
                task.hoi_refs[motion, slot, frame] = state[
                    source, source_frame
                ]
    task.ref_reward[:, 1:, :] = (
        task.ref_reward[:, 1:, :] * (1 - 1e-5)
    )


def _task(seed: int):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    envs, motions, frames, slots, features = 8, 2, 96, 4, 7
    starts = torch.tensor(
        [0, 3, 7, 11, 1, 5, 9, 13],
        device=device,
    )
    ends = starts + torch.tensor(
        [31, 40, 50, 60, 35, 45, 55, 63],
        device=device,
    )
    data_id = torch.arange(envs, device=device) % motions
    reset = torch.ones(envs, dtype=torch.long, device=device)
    task = SimpleNamespace(
        reset_buf=reset,
        _terminate_buf=torch.zeros_like(reset),
        progress_buf=ends,
        obs_buf=torch.zeros((envs, 1), device=device),
        _rigid_body_pos=torch.zeros((envs, 1, 3), device=device),
        max_episode_length=torch.full(
            (motions,), frames, dtype=torch.long, device=device
        ),
        data_id=data_id,
        _enable_early_termination=True,
        _termination_heights=torch.zeros(1, device=device),
        start_times=starts,
        rollout_length=64,
        kinematic_reset=torch.zeros(envs, dtype=torch.bool, device=device),
        contact_reset=torch.zeros((envs, 1), device=device),
        enable_evaluation=False,
        psi=4,
        _reference_update_possible=True,
        _all_env_ids=torch.arange(envs, device=device),
        _sum_reward=torch.rand(
            envs, generator=generator, device=device
        ),
        _curr_reward=torch.rand(
            (envs, frames), generator=generator, device=device
        ),
        _curr_state=torch.rand(
            (envs, frames, features),
            generator=generator,
            device=device,
        ),
        hoi_refs=torch.rand(
            (motions, slots, frames, features),
            generator=generator,
            device=device,
        ),
        ref_reward=torch.rand(
            (motions, slots, frames),
            generator=generator,
            device=device,
        ),
    )
    task.compute_hoi_reset = lambda *args: (
        reset.clone(),
        torch.zeros_like(reset),
    )
    return task


def _clone(task):
    values = {
        name: value.clone() if torch.is_tensor(value) else value
        for name, value in vars(task).items()
    }
    clone = SimpleNamespace(**values)
    reset = clone.reset_buf
    clone.compute_hoi_reset = lambda *args: (
        reset.clone(),
        torch.zeros_like(reset),
    )
    return clone


def _check_single_motion_sampling() -> None:
    env_ids = torch.arange(257, device="cuda")
    obj2motion = torch.ones((1, 1), dtype=torch.bool, device="cuda")
    task = SimpleNamespace(
        num_motions=1,
        device="cuda",
        object_name=["object"],
        obj2motion=obj2motion,
        _single_motion_fixed_start=True,
    )
    for seed in range(4):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        old_ids = torch.stack(
            [
                torch.where(obj2motion[0])[0][
                    torch.randint(obj2motion[0].sum(), ())
                ]
                for _ in env_ids
            ]
        ).to("cuda")
        old_mask = torch.bernoulli(
            torch.full((len(env_ids),), 0.1, device="cuda")
        ).bool()
        reset_ids = env_ids[old_mask]
        old_times = torch.cat(
            [
                torch.searchsorted(
                    torch.ones(1, device="cuda"),
                    torch.rand(1).to("cuda"),
                )
                if env_id not in reset_ids
                else torch.zeros(1, device="cuda", dtype=torch.long)
                for env_id in env_ids
            ]
        )
        old_cpu_state = torch.get_rng_state()
        old_cuda_state = torch.cuda.get_rng_state()

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        new_ids = InterMimic._sample_motion_ids(task, env_ids)
        new_mask = torch.bernoulli(
            torch.full((len(env_ids),), 0.1, device="cuda")
        ).bool()
        new_times = InterMimic._sample_hybrid_motion_times(
            task, new_ids, env_ids, new_mask
        )
        if not torch.equal(old_ids, new_ids):
            raise AssertionError("single-motion IDs changed")
        if not torch.equal(old_mask, new_mask):
            raise AssertionError("hybrid mask changed")
        if not torch.equal(old_times, new_times):
            raise AssertionError("fixed-start samples changed")
        if not torch.equal(old_cpu_state, torch.get_rng_state()):
            raise AssertionError("CPU RNG state changed")
        if not torch.equal(old_cuda_state, torch.cuda.get_rng_state()):
            raise AssertionError("CUDA RNG state changed")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    for seed in range(8):
        source = _task(seed)
        old = _clone(source)
        new = _clone(source)
        torch.manual_seed(seed + 1000)
        _old_update(old)
        torch.manual_seed(seed + 1000)
        InterMimic._compute_reset(new)
        for name in (
            "_sum_reward",
            "_curr_reward",
            "ref_reward",
            "hoi_refs",
        ):
            if not torch.equal(getattr(old, name), getattr(new, name)):
                raise AssertionError(f"{name} mismatch at seed {seed}")
    source = _task(100)
    source.max_episode_length[:] = source.rollout_length
    source._reference_update_possible = False
    old = _clone(source)
    new = _clone(source)
    _old_update(old)
    InterMimic._compute_reset(new)
    for name in (
        "_sum_reward",
        "_curr_reward",
        "ref_reward",
        "hoi_refs",
    ):
        if not torch.equal(getattr(old, name), getattr(new, name)):
            raise AssertionError(f"{name} no-update fast-path mismatch")
    _check_single_motion_sampling()
    points1 = torch.randn((5, 17, 3), device="cuda")
    points2 = torch.randn((5, 11, 3), device="cuda")
    points2[:, 1] = points2[:, 0]
    distances = points1.unsqueeze(2) - points2.unsqueeze(1)
    nearest = torch.argmin(torch.norm(distances, dim=-1), dim=-1)
    batch, point = torch.meshgrid(
        torch.arange(points1.shape[0]),
        torch.arange(points1.shape[1]),
        indexing="ij",
    )
    expected_sdf = distances[batch, point, nearest].contiguous()
    if not torch.equal(expected_sdf, compute_sdf(points1, points2)):
        raise AssertionError("compute_sdf gather mismatch")
    old_curr = torch.randn((32, 97), device="cuda")
    old_hist = torch.randn_like(old_curr)
    new_curr = old_curr.clone()
    new_hist = old_hist.clone()
    for step in range(16):
        if step % 3 == 0:
            rows = torch.arange(step % 7, 32, 7, device="cuda")
            old_hist[rows] = 0
            new_hist[rows] = 0
        old_hist = old_curr.clone()
        new_hist, new_curr = new_curr, new_hist
        next_obs = torch.randn_like(old_curr)
        old_curr[:] = next_obs
        new_curr[:] = next_obs
        if not torch.equal(old_hist, new_hist):
            raise AssertionError(f"history buffer mismatch at step {step}")
        if not torch.equal(old_curr, new_curr):
            raise AssertionError(f"current buffer mismatch at step {step}")
    print("InterMimic reference update: exact CUDA tensor parity")
    print("InterMimic no-update fast path: exact CUDA tensor parity")
    print("InterMimic single-motion sampling: exact RNG/tensor parity")
    print("InterMimic SDF/history buffers: exact CUDA tensor parity")


if __name__ == "__main__":
    main()
