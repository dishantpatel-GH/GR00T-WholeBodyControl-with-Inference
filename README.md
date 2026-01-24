# gr00t_wbc

Software stack for loco-manipulation experiments across multiple humanoid platforms, with primary support for the Unitree G1. This repository provides whole-body control policies, and inference stack. 

---

## System Installation

### Repository Setup
Install Git and Git LFS:
```bash
sudo apt update
sudo apt install git git-lfs
git lfs install
```

Clone the repository:
```bash
mkdir -p ~/Projects
cd ~/Projects
git clone https://github.com/dishantpatel-GH/GR00T-WholeBodyControl-with-Inference.git
cd GR00T-WholeBodyControl-with-Inference
```

### Conda Environment

Install a conda environment and with all the dependencies:
```bash
./install_minimal_real_robot.sh
```
This creates conda environment `gr00t_robot`.

Start or re-enter a conda environment:
```bash
conda activate gr00t_robot
```

---

## Evaluate checkpoint

### Terminal 1 - Server:

```bash
python gr00t/eval/run_gr00t_server.py \
  --model-path <checkpoint_path> \
  --embodiment-tag UNITREE_G1 \
  --device cuda \
  --host 0.0.0.0 \
  --port 5555
```

### Terminal 2 - Client:

Run `run_g1_control_loop.py`:
```bash
python gr00t_wbc/control/main/teleop/run_g1_control_loop.py \
  --interface eno1 \
  --control_frequency 50 \
  --no-with_hands
```

Run `run_gr00t_inference_policy.py`:
```bash
python scripts/run_gr00t_inference_policy.py \
  --model_host <server ip> \
  --model_port 5555 \
  --camera_host <robot ip> \
  --camera_port 5556 \
  --task_description "task name" \
  --inference_frequency 30 \
  --n_action_steps 25 \
  --no-with_hands
```
