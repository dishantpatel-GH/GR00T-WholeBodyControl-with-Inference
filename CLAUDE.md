# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository combines two systems for humanoid robot manipulation:

1. **gr00t_wbc** — Whole-body control stack for the Unitree G1 robot (low-level control, kinematics, motor commands)
2. **Isaac-GR00T** — NVIDIA's GR00T N1.6 Vision-Language-Action (VLA) foundation model (3B params, PyTorch/HF Trainer)

Isaac-GR00T predicts robot actions from vision+language inputs. gr00t_wbc handles low-level execution on real hardware. The `gr00t/` package at root provides a lightweight inference bridge (policy server/client, data types, embodiment configs) that connects the two.

## Setup & Installation

### gr00t_wbc (robot control — conda, Python 3.10)
```bash
./install_minimal_real_robot.sh        # creates conda env "gr00t_robot" (minimal, real robot only)
conda activate gr00t_robot
pip install -e ".[dev]"                # dev extras (pytest, ruff, black)
```
The env name defaults to `gr00t_robot` but can be overridden via `GR00T_ENV_NAME`.

### Isaac-GR00T (VLA model — uv, Python 3.10)
```bash
cd Isaac-GR00T
uv sync --python 3.10
uv pip install -e .
# Docker alternative:
bash docker/build.sh && docker run -it --rm --gpus all gr00t-dev /bin/bash
```

## Key Commands

### Linting & Formatting (root project)
```bash
ruff check .                    # lint (E, F, I rules)
ruff check --fix .              # auto-fix
black --check .                 # format check (line-length 100)
black .                         # auto-format
mypy gr00t_wbc/                 # type check
```

### Tests
```bash
cd Isaac-GR00T && uv run pytest tests/
```
No `tests/` directory exists at root level currently.

### Fine-tuning — Isaac-GR00T
```bash
cd Isaac-GR00T
CUDA_VISIBLE_DEVICES=0 uv run python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path <DATASET_PATH> \
    --embodiment-tag UNITREE_G1 \
    --num-gpus 1 --output-dir <OUTPUT_PATH> --max-steps 2000 --global-batch-size 32

# G1-specific example
uv run bash examples/GR00T-WholeBodyControl/finetune_g1.sh
```

### Evaluation — Isaac-GR00T
```bash
cd Isaac-GR00T
uv run python gr00t/eval/open_loop_eval.py \
    --dataset-path <PATH> --embodiment-tag UNITREE_G1 --model-path <CKPT>
```

### Running on Real Robot (3 terminals)

**Terminal 1 — GR00T Inference Server** (GPU machine):
```bash
cd Isaac-GR00T
uv run --extra=gpu python gr00t/eval/run_gr00t_server.py \
  --model-path <checkpoint> --embodiment-tag UNITREE_G1 \
  --device cuda --host 0.0.0.0 --port 5555
```

**Terminal 2 — Robot Control Loop** (robot or connected machine):
```bash
python gr00t_wbc/control/main/teleop/run_g1_control_loop.py \
  --interface eno1 --control_frequency 50 --no-with_hands
```

**Terminal 3 — Inference Policy Client** (bridges server ↔ control loop):
```bash
python scripts/run_gr00t_inference_policy.py \
  --model_host <server_ip> --model_port 5555 \
  --camera_host <robot_ip> --camera_port 5556 \
  --task_description "task" --inference_frequency 30 --n_action_steps 25 --no-with_hands
```

## Architecture

### End-to-End Data Flow

```
[Training / Fine-tuning]
  Teleoperation data (LeRobot v2) → Isaac-GR00T training → Checkpoint

[Real Robot Inference]
  GR00T Server (ZMQ:5555)  ←→  Inference Policy Client  →(ROS topic)→  Control Loop  →(DDS)→  Robot
                                 ↑ camera observations (ZMQ from camera server, port 5556)
```

### `gr00t/` (root-level inference bridge)

Lightweight package connecting Isaac-GR00T model serving to the robot control stack:

- `policy/policy.py` — `BasePolicy` abstract class
- `policy/server_client.py` — `PolicyServer`/`PolicyClient` (ZMQ + msgpack serialization)
- `data/embodiment_tags.py` — Robot type definitions (`EmbodimentTag`)
- `data/types.py` — `ModalityConfig` and data type definitions
- `configs/` — Embodiment and data configuration files

### Isaac-GR00T (`Isaac-GR00T/gr00t/`)

VLA foundation model — PyTorch, HuggingFace Trainer, DeepSpeed:

- **`model/gr00t_n1d6/`** — VLM backbone (Cosmos-Reason-2B) → 32-layer Diffusion Transformer → action output
- **`model/modules/`** — `eagle_backbone.py` (VLM), `dit.py` (DiT/AlternateVLDiT), action head
- **`data/dataset/`** — `ShardedSingleStepDataset`, `ShardedMixtureDataset`, `LeRobotEpisodeLoader`
- **`data/embodiment_tags.py`** — Robot type definitions (UNITREE_G1, GR1, LIBERO_PANDA, etc.)
- **`experiment/`** — `launch_finetune.py` (simple), `launch_train.py` (advanced)
- **`policy/`** — `gr00t_policy.py` (inference wrapper), `server_client.py` (ZMQ serving)
- **`eval/`** — Sim benchmarks (LIBERO, BEHAVIOR, RoboCasa, G1 LocoManipulation)
- **`configs/`** — Model, embodiment, DeepSpeed, training configs

### gr00t_wbc (`gr00t_wbc/`)

Low-level whole body control for Unitree G1:

- **`control/base/`** — Abstract interfaces: `Env`, `HumanoidEnv`, `Policy`, `Sensor`
- **`control/envs/`** — Robot environment implementations (G1: observe/queue_action)
- **`control/policy/`** — `g1_decoupled_whole_body_policy.py` (upper+lower body), `g1_gear_wbc_policy.py` (ONNX locomotion), `interpolation_policy.py`, `wbc_policy_factory.py`
- **`control/robot_model/`** — Pinocchio wrapper for FK/IK
- **`control/sensor/`** — Sensor interfaces
- **`control/main/teleop/run_g1_control_loop.py`** — Main control loop entry point
- **`control/main/constants.py`** — ROS topics, default poses, control constants
- **`sim2mujoco/`** — MuJoCo simulation utilities

### Key Design Patterns

- **Decoupled control**: Upper body (IK from policy) + lower body (ONNX locomotion) combined by `G1DecoupledWholeBodyPolicy`
- **Embodiment-conditioned**: Isaac-GR00T handles multiple robots via `EmbodimentTag`
- **Data format**: LeRobot v2 (parquet + mp4 videos)
- **Inference optimization**: Isaac-GR00T supports TensorRT/ONNX export
- **Config-driven**: Uses `tyro` for CLI arg parsing (dataclass-based), `hydra-core` for config management

### Communication Stack

| Layer | Protocol | Purpose |
|-------|----------|---------|
| Model inference | ZMQ + msgpack | PolicyServer ↔ PolicyClient (port 5555) |
| Camera stream | ZMQ + pickle | Camera server → Inference client (port 5556) |
| Process coordination | ROS2 topics | Control loop ↔ Inference policy |
| Robot hardware | DDS (Unitree SDK) | Motor commands, joint state, IMU |

## Code Style

- **black**: line-length 100, **ruff**: line-length 115, target py310, rules E/F/I
- **isort**: black-compatible profile
- Exclude `external_dependencies/` and `gr00t_wbc/dexmg/` from all linting
- `__init__.py` files exempt from F401 (unused imports)
- Root project: conda + pip + Python 3.10 | Isaac-GR00T: `uv` + Python 3.10
