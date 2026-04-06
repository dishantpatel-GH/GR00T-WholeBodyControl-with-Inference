"""
Compare inference debug logs against dataset distribution.

Loads the saved observations/actions from --debug_dir and compares them
against the dataset at --dataset_path. Prints per-joint statistics and
flags values that fall outside the dataset's [min, max] range.

Usage:
    python scripts/compare_debug_to_dataset.py \
        --debug_dir logs_debug \
        --dataset_path pick_place_biscuit \
        --episode -1
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tyro


@dataclass
class CompareConfig:
    debug_dir: str = "logs_debug"
    """Directory with debug logs from inference"""
    dataset_path: str = "pick_place_biscuit"
    """Dataset to compare against"""
    episode: int = -1
    """Episode index (-1 = use ALL episodes)"""


BODY_PARTS = ["left_leg", "right_leg", "waist",
              "left_arm", "right_arm", "left_hand", "right_hand"]

ACTION_PARTS = ["left_arm", "right_arm", "left_hand", "right_hand", "waist"]


def load_dataset_stats(dataset_path: str, episode: int) -> dict:
    """Load dataset actions/states and compute per-joint statistics."""
    ds_dir = Path(dataset_path)
    if not ds_dir.is_absolute():
        if not ds_dir.exists():
            project_root = Path(__file__).resolve().parent.parent
            ds_dir = project_root / dataset_path

    with open(ds_dir / "meta" / "info.json") as f:
        info = json.load(f)
    with open(ds_dir / "meta" / "modality.json") as f:
        modality = json.load(f)

    slices = {}
    for part in BODY_PARTS:
        entry = modality["state"][part]
        slices[part] = (entry["start"], entry["end"])

    joint_names = info["features"]["observation.state"]["names"]
    total_episodes = info["total_episodes"]
    chunks_size = info.get("chunks_size", 1000)
    data_template = info["data_path"]

    # Load episodes
    if episode >= 0:
        episodes = [episode]
    else:
        episodes = list(range(total_episodes))

    all_states = []
    all_actions = []
    for ep_idx in episodes:
        chunk = ep_idx // chunks_size
        path = ds_dir / data_template.format(episode_chunk=chunk, episode_index=ep_idx)
        if not path.exists():
            continue
        table = pq.read_table(path)
        states = np.array([r.as_py() for r in table.column("observation.state")], dtype=np.float64)
        actions = np.array([r.as_py() for r in table.column("action")], dtype=np.float64)
        all_states.append(states)
        all_actions.append(actions)

    all_states = np.concatenate(all_states)
    all_actions = np.concatenate(all_actions)

    print(f"Dataset: {len(episodes)} episodes, {len(all_states)} frames, {all_states.shape[1]}D")
    return {
        "states": all_states,
        "actions": all_actions,
        "slices": slices,
        "joint_names": joint_names,
    }


def print_comparison(label: str, debug_data: np.ndarray, dataset_data: np.ndarray,
                     joint_names: list, slices: dict):
    """Print per-body-part comparison."""
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"  Debug: {debug_data.shape[0]} samples, Dataset: {dataset_data.shape[0]} samples")
    print(f"{'='*80}")

    dim = min(debug_data.shape[1], dataset_data.shape[1])

    for part_name in BODY_PARTS:
        if part_name not in slices:
            continue
        s, e = slices[part_name]
        if s >= dim:
            continue
        e = min(e, dim)

        print(f"\n  --- {part_name} [{s}:{e}] ---")
        print(f"  {'Joint':<40s} {'Debug min':>10s} {'Debug max':>10s} "
              f"{'DS min':>10s} {'DS max':>10s} {'DS mean':>10s} {'OOD?':>6s}")

        for j in range(s, e):
            name = joint_names[j] if j < len(joint_names) else f"joint_{j}"
            d_min = debug_data[:, j].min()
            d_max = debug_data[:, j].max()
            ds_min = dataset_data[:, j].min()
            ds_max = dataset_data[:, j].max()
            ds_mean = dataset_data[:, j].mean()

            # Check out-of-distribution
            margin = (ds_max - ds_min) * 0.1  # 10% margin
            ood = ""
            if d_min < ds_min - margin or d_max > ds_max + margin:
                ood = " !! OOD"

            print(f"  {name:<40s} {d_min:10.4f} {d_max:10.4f} "
                  f"{ds_min:10.4f} {ds_max:10.4f} {ds_mean:10.4f}{ood}")


def main(config: CompareConfig):
    debug_dir = Path(config.debug_dir)

    # Load debug data
    obs_path = debug_dir / "obs_states.npy"
    act_path = debug_dir / "actions_41d.npy"
    raw_path = debug_dir / "raw_actions.npz"

    if not obs_path.exists() and not act_path.exists():
        print(f"No debug data found in {debug_dir}/")
        print("Run inference with --debug_log first.")
        return

    # Load dataset stats
    ds = load_dataset_stats(config.dataset_path, config.episode)

    # Compare observations (states sent to model)
    if obs_path.exists():
        obs_states = np.load(obs_path)
        print_comparison(
            "OBSERVATION STATE (sent to model, hands normalized 0-1)",
            obs_states, ds["states"], ds["joint_names"], ds["slices"],
        )
    else:
        print("No obs_states.npy found, skipping observation comparison.")

    # Compare actions (model output in dataset format)
    if act_path.exists():
        actions_41d = np.load(act_path)
        print_comparison(
            "ACTION (model output, first timestep, hands normalized 0-1)",
            actions_41d, ds["actions"], ds["joint_names"], ds["slices"],
        )
    else:
        print("No actions_41d.npy found, skipping action comparison.")

    # Print raw action shapes and ranges
    if raw_path.exists():
        raw = np.load(raw_path)
        print(f"\n{'='*80}")
        print(f"  RAW MODEL OUTPUT (per-key)")
        print(f"{'='*80}")
        for key in sorted(raw.files):
            arr = raw[key]
            print(f"  {key:<25s} shape={str(arr.shape):<15s} "
                  f"min={arr.min():10.4f} max={arr.max():10.4f} "
                  f"mean={arr.mean():10.4f} std={arr.std():10.4f}")

    print("\nDone.")


if __name__ == "__main__":
    config = tyro.cli(CompareConfig)
    main(config)
