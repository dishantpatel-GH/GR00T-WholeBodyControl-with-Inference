> # Grasp Verification Pipeline — Complete Implementation Guide

## Overview

A single lightweight model that takes wrist camera frames + Inspire hand motor currents as input and outputs grasp success/failure. Runs on the **GPU server** alongside GR00T (not on the robot) to minimize robot-side compute. Verification only happens **once after initial grasp close**, not during transport.

***

## Part 1: System Architecture — What Runs Where

```
ROBOT (Jetson Orin) — keep load minimal          GPU SERVER — has spare capacity
──────────────────────────────────────           ──────────────────────────────────

Control Loop (50Hz)                              GR00T Model (runs on demand)
  ├─ reads tau_est from Inspire (already does)     ↑
  ├─ publishes obs on STATE_TOPIC (already does)   │ ZMQ :5555 "get_action"
  └─ executes WBC actions                         │
                                                  │
Camera Server (30Hz)                              Grasp Verifier Model ← NEW
  ├─ streams ego_view (already does)               │ (same process, loaded once)
  ├─ streams left_wrist (configure to enable)      │
  └─ streams right_wrist (configure to enable)     │ ZMQ :5555 "verify_grasp" ← NEW endpoint
                                                  │
Inference Policy Client (50Hz)                    │
  ├─ queries "get_action" (already does)  ────────┘
  ├─ queries "verify_grasp" (NEW) ────────────────┘
  ├─ state machine: EXECUTE → VERIFY → RETRY
  └─ lightweight: just sends data, receives bool
```

**Why this works:**

* Robot only sends a wrist image (JPEG, \~30KB) + 6 floats (48 bytes) over ZMQ — negligible

* Server already has the GPU loaded with GR00T. The grasp classifier is \~4MB (MobileNetV3-Small). Loading it alongside a 3B param model adds nothing

* Inference takes \~2ms on GPU vs \~50ms on Orin CPU — 25x faster on server

* No new connections — reuses the existing ZMQ PolicyServer on port 5555, just adds one endpoint

***

## Part 2: The Model

### Why MobileNetV3-Small (not ResNet-18)

| Model                 | Params   | GPU inference | Accuracy (expected) |
| --------------------- | -------- | ------------- | ------------------- |
| ResNet-18             | 11.7M    | \~3ms         | \~94%               |
| **MobileNetV3-Small** | **2.5M** | **\~1.5ms**   | **\~93%**           |
| EfficientNet-B0       | 5.3M     | \~4ms         | \~95%               |

MobileNetV3-Small is the right choice because:

* The task is simple (binary classification of a 224x224 image + 6 scalars)

* 2.5M params vs GR00T's 3B — rounds to zero additional memory

* 1.5ms inference means verification adds no meaningful latency to the control loop

* Accuracy difference vs larger models is <2% for this task

### Model Architecture

```
Input 1: Wrist camera image (224 × 224 × 3, uint8)
Input 2: Finger motor currents (6 floats — tau_est from Inspire hand)

┌─────────────────────────────────────────┐
│ MobileNetV3-Small backbone (pretrained) │
│   Input: (B, 3, 224, 224)              │
│   Output: (B, 576) feature vector       │
└──────────────┬──────────────────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Concatenate: [576] + [6]     │
│   = (B, 582) combined vector │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Linear(582, 128) → ReLU      │
│ Dropout(0.3)                 │
│ Linear(128, 1) → Sigmoid     │
│   Output: probability [0, 1] │
└──────────────────────────────┘

Threshold: p > 0.5 → HOLDING
           p ≤ 0.5 → NOT HOLDING (trigger retry)
```

### Why include motor currents in the model (not just vision)?

The model learns the **joint signal** between what the camera sees and what the fingers feel:

| Scenario                                  | Camera alone           | Camera + tau\_est          |
| ----------------------------------------- | ---------------------- | -------------------------- |
| Object in shadow, fingers closed          | Ambiguous (dark image) | tau says contact → HOLDING |
| Transparent object (glass)                | Sees nothing           | tau says contact → HOLDING |
| Object color matches hand                 | Can't segment          | tau says contact → HOLDING |
| Foam/soft object (no deformation visible) | Looks empty            | tau > 0 → HOLDING          |
| Hand is covered with dust/grime           | Sees "something"       | tau ≈ 0 → NOT HOLDING      |

The 6 tau values cost nothing (48 bytes) and resolve the cases where vision alone is ambiguous. The model learns when to trust vision vs when to trust torque.

***

## Part 3: Creating the Dataset

### 3.1 What data we need

```
Per sample:
  - wrist_image: 224×224×3 RGB (from left_wrist or right_wrist camera)
  - tau_est: 6 floats (from Inspire hand, same side as camera)
  - label: 0 (NOT HOLDING) or 1 (HOLDING)
```

Target: **\~500 samples minimum** (250 positive, 250 negative). More is better — 1000+ samples will push accuracy above 95%.

### 3.2 How to collect it

We don't need a separate data collection run. We modify the **existing teleoperation pipeline** to save verification frames at the right moment.

**Collection script:** **`scripts/data_collection/collect_grasp_data.py`**

This script connects to the same ROS topics and camera server that the inference policy uses. It runs as a **4th terminal** during normal teleoperation:

```
Terminal 1: GR00T server (or teleop leader)
Terminal 2: Control loop
Terminal 3: Teleop/inference policy (operator does tasks normally)
Terminal 4: collect_grasp_data.py ← NEW (passive observer)
```

How it works:

```
STEP 1: Monitor finger commands
  - Subscribe to CONTROL_GOAL_TOPIC
  - Subscribe to STATE_TOPIC (for tau_est)
  - Connect to camera server (for wrist images)
  - Watch for grasp close command (hand joint targets → closed position)

STEP 2: When grasp close detected
  - Wait 0.5 seconds (let fingers settle)
  - Capture: wrist_image + tau_est at that moment
  - Beep / print to console: "GRASP DETECTED — press 'y' for success, 'n' for failure"

STEP 3: Human labels
  - Operator presses 'y' or 'n'
  - Save {wrist_image, tau_est, label} to dataset directory

STEP 4: Auto-negative collection
  - Every 30 seconds during non-grasp periods, save a frame with label=0
  - Captures: empty hand, hand near objects but not grasping, hand in transit
  - These auto-negatives prevent the model from learning "any image = holding"
```

### 3.3 Dataset directory structure

```
grasp_verification_data/
├── images/
│   ├── 000000.jpg          # wrist camera frame
│   ├── 000001.jpg
│   └── ...
├── metadata.csv            # tau_est + labels
│   ┌──────────┬────────┬────────┬────────┬────────┬────────┬────────┬───────┐
│   │ image_id │ tau_0  │ tau_1  │ tau_2  │ tau_3  │ tau_4  │ tau_5  │ label │
│   ├──────────┼────────┼────────┼────────┼────────┼────────┼────────┼───────┤
│   │ 000000   │ 0.35   │ 0.28   │ 0.31   │ 0.12   │ 0.05   │ 0.00   │ 1     │
│   │ 000001   │ 0.02   │ 0.01   │ 0.01   │ 0.01   │ 0.01   │ 0.00   │ 0     │
│   └──────────┴────────┴────────┴────────┴────────┴────────┴────────┴───────┘
└── splits/
    ├── train.txt           # 80% of image_ids
    └── val.txt             # 20% of image_ids
```

### 3.4 Data augmentation (at training time, not collection time)

```python
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),       # hand can be mirrored
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),  # simulate camera shake
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# tau_est augmentation: add small Gaussian noise (σ=0.02) to simulate sensor noise
tau_augmented = tau_est + np.random.normal(0, 0.02, size=6)
```

### 3.5 What to collect in each category

**Positive samples (label=1, HOLDING):**

* Grasp on mug, bottle, box, ball, plate, utensils

* Different lighting (bright, dim, side-lit)

* Different hand orientations (top grasp, side grasp, pinch)

* Partial occlusion (object partially hidden by fingers)

* Different object sizes (small pen vs large box)

**Negative samples (label=0, NOT HOLDING):**

* Fingers closed on air (most common failure)

* Hand open near object (pre-grasp)

* Hand in transit (no object)

* Object near hand but not in fingers (slipped)

* Object on table, hand above it (missed grasp)

* Random frames during walking/standing (auto-collected)

### 3.6 Hard negatives (collect deliberately)

These are the cases that fool a simple classifier:

* Fingers closed around a thin object that barely shows in camera → label=1

* Fingers closed on table edge (looks like grasping) → label=0

* Object pressed against palm but fingers not wrapped → label=0

* Two objects, one in hand one on table → label=1

Collect \~50 hard negatives deliberately by staging these scenarios during a separate collection session.

***

## Part 4: Training the Model

### 4.1 Training script: `scripts/training/train_grasp_verifier.py`

Run on the GPU server (same machine that runs GR00T inference):

```bash
python scripts/training/train_grasp_verifier.py \
    --data_dir grasp_verification_data/ \
    --output_dir checkpoints/grasp_verifier/ \
    --epochs 30 \
    --batch_size 32 \
    --lr 1e-4 \
    --pretrained  # use ImageNet-pretrained MobileNetV3-Small
```

### 4.2 Training details

```
Model:        MobileNetV3-Small (torchvision.models.mobilenet_v3_small, pretrained=True)
Head:         Replace classifier with: Linear(576+6, 128) → ReLU → Dropout(0.3) → Linear(128, 1)
Loss:         Binary Cross-Entropy with Logits (BCEWithLogitsLoss)
Optimizer:    AdamW, lr=1e-4, weight_decay=1e-4
Scheduler:    CosineAnnealingLR, T_max=30
Batch size:   32
Epochs:       30 (converges fast — simple task)
Backbone:     Freeze first 8 layers for first 5 epochs, then unfreeze all
Data split:   80% train, 20% validation (stratified by label)
```

### 4.3 Expected training time

* \~500 samples, 30 epochs, batch size 32 → \~470 steps/epoch → \~14,100 total steps

* At ~50ms/step on a single GPU → **\~12 minutes total training**

* With 1000 samples → \~24 minutes

### 4.4 What metrics to track

```
Primary:    Accuracy (target: >93%)
Critical:   False Negative Rate (model says HOLDING but actually NOT)
            → This is the dangerous one — robot thinks it has the object and proceeds
            → Target: <3% false negative rate
Secondary:  False Positive Rate (model says NOT HOLDING but actually is)
            → This wastes a retry but doesn't cause task failure
            → Target: <10% is acceptable
```

Prioritize recall (catching real failures) over precision. A false retry is cheap. A missed failure cascades.

### 4.5 Model export

```python
# After training, export to TorchScript for fast loading
model = GraspVerifier.load_from_checkpoint("checkpoints/grasp_verifier/best.pt")
model.eval()

# Trace with example inputs
example_image = torch.randn(1, 3, 224, 224)
example_tau = torch.randn(1, 6)
traced = torch.jit.trace(model, (example_image, example_tau))
traced.save("checkpoints/grasp_verifier/grasp_verifier.pt")
```

TorchScript model loads in \~100ms and runs without Python overhead.

***

## Part 5: Deployment — Integrating into the System

### 5.1 Files that change

| File                                                    | Change                                                                                                         | Why                                      |
| ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| `Isaac-GR00T/gr00t/eval/run_gr00t_server.py`            | Load grasp verifier model at startup, register `verify_grasp` endpoint                                         | Server-side: add verification capability |
| `scripts/run_gr00t_inference_policy.py`                 | Add state machine (EXECUTE→VERIFY→RETRY), call `verify_grasp` endpoint after grasp, retract + retry on failure | Client-side: use verification            |
| `gr00t_wbc/control/sensor/composed_camera.py`           | No code change — just enable wrist cameras in config when launching camera server                              | Hardware: enable wrist cam stream        |
| **NEW** `scripts/verification/grasp_verifier.py`        | Model definition + inference wrapper                                                                           | The classifier itself                    |
| **NEW** `scripts/data_collection/collect_grasp_data.py` | Dataset collection tool                                                                                        | Data pipeline                            |
| **NEW** `scripts/training/train_grasp_verifier.py`      | Training script                                                                                                | Model training                           |

### 5.2 Server-side changes: `run_gr00t_server.py`

The GR00T server already supports custom endpoints via `register_endpoint()`. We add one:

```python
# In run_gr00t_server.py, after policy is created:

# Load grasp verifier (tiny model, shares GPU with GR00T)
grasp_verifier = GraspVerifierModel("checkpoints/grasp_verifier/grasp_verifier.pt", device="cuda")

def handle_verify_grasp(**kwargs):
    """Endpoint: takes wrist image + tau_est, returns {holding: bool, confidence: float}"""
    wrist_image = kwargs["wrist_image"]   # (H, W, 3) uint8
    tau_est = kwargs["tau_est"]           # (6,) float
    holding, confidence = grasp_verifier.predict(wrist_image, tau_est)
    return {"holding": bool(holding), "confidence": float(confidence)}

server.register_endpoint("verify_grasp", handle_verify_grasp, requires_input=True)
```

Cost on server: \~4MB VRAM for model weights. Inference: \~1.5ms per call. The GR00T model uses \~6GB — this adds 0.07%.

### 5.3 Client-side changes: `run_gr00t_inference_policy.py`

The main loop currently runs linearly. It becomes a state machine:

```
EXECUTING      → buffer runs out after grasp command   → VERIFYING
EXECUTING      → buffer runs out (no grasp)            → EXECUTING (normal re-query)
VERIFYING      → server says HOLDING                   → EXECUTING (continue task)
VERIFYING      → server says NOT HOLDING               → RESETTING
RESETTING      → level 1/2/3 reset complete            → RETRYING
RETRYING       → attempt < 3                           → EXECUTING (re-query with fresh obs)
RETRYING       → attempt ≥ 3                           → ABORTING
ABORTING       → hold position, log failure
```

**Detecting "was this a grasp action?"**
The model outputs Inspire hand joint targets. If the target goes from open → closed (delta exceeds threshold), that's a grasp command. We track this:

```python
def is_grasp_command(self, current_action, previous_action):
    """Check if action transitions from open to closed hand."""
    if previous_action is None:
        return False
    # hand joints are in the action dict, compare finger targets
    curr_hand = current_action["target_upper_body_pose"][self.hand_indices]
    prev_hand = previous_action["target_upper_body_pose"][self.hand_indices]
    # if fingers moved significantly toward closed position
    delta = curr_hand - prev_hand
    return np.mean(delta) > GRASP_CLOSE_THRESHOLD
```

**When verification runs:**

* Only once, after the action buffer empties following a detected grasp command

* NOT during transport — the model keeps executing normally

* NOT on every action step — only at chunk boundaries when a grasp was in the chunk

**Robot-side cost of verification:**

```
1. Read wrist image from camera server     → already streaming, ~0 extra cost
2. Read tau_est from state                 → already in obs dict, ~0 extra cost
3. Send image + tau over ZMQ to server     → ~30KB JPEG + 48 bytes, ~2ms network
4. Receive response                        → 1 bool + 1 float, ~0.1ms
5. Total robot-side cost: ~2ms per verification call
6. Happens once per grasp attempt, not per control loop tick
```

### 5.4 How reset works (only after initial grasp failure)

**Key principle: Do NOT retract to home.** The arm is already near the object. Going home wastes 3-5 seconds and forces the model to re-plan the entire approach. Instead, use the minimum reset needed.

#### Three reset levels — chosen automatically based on failure type

```
LEVEL 1: "Missed grasp — object still in place" (~0.7s)
  Fingers closed on air, no contact detected.
  → Open Inspire fingers (0.5s)
  → Do NOT move arm at all — it's already positioned
  → Re-query GR00T immediately with fresh image
  → Model sees: hand open, near object → plans short re-grasp

LEVEL 2: "Grasp disturbed object — it shifted/tipped" (~1.5s)
  Fingers had contact but object wasn't secured.
  → Open Inspire fingers (0.5s)
  → Pull arm back slightly: blend 80% current pose + 20% home pose (0.5s)
  → Wait 0.3s for camera to settle
  → Re-query GR00T with fresh image
  → Model sees: hand open, slightly back, object in new position → adjusts approach

LEVEL 3: "Arm in unsafe pose" (~3.0s) — SAFETY FALLBACK ONLY
  Arm near joint limits or abnormally high torques.
  → Open Inspire fingers (0.5s)
  → Retract arm fully to home pose (2.0s)
  → Wait 0.5s
  → Re-query GR00T
  → Used rarely — only when continuing from current pose is unsafe
```

#### How level is chosen

```python
def choose_reset_level(self, state, verify_result):
    """Pick minimum reset needed based on what went wrong."""
    arm_tau = state["tau_est"][self.arm_indices]
    arm_q = state["q"][self.arm_indices]

    # Level 3: arm safety issue (near joint limits or high torque)
    if self.near_joint_limits(arm_q) or np.any(np.abs(arm_tau) > ARM_TAU_DANGER):
        return 3

    # Level 1 vs 2: did we disturb the object?
    # If Inspire fingers had ANY contact (tau > 0), object probably shifted
    finger_tau = state["tau_est"][self.hand_indices[:6]]  # 6 Inspire motors
    had_contact = np.any(finger_tau > TAU_CONTACT_THRESHOLD)

    if had_contact:
        return 2  # slight pullback — object likely moved
    else:
        return 1  # fingers only — closed on air, object untouched
```

#### Reset actions

**Level 1 — fingers only (most common, fastest):**
```python
def reset_level_1(self, state):
    """Open Inspire fingers, keep arm exactly where it is."""
    action = self._get_hold_position_action(state["q"])  # already exists
    # Override just the hand joints to open
    action["target_upper_body_pose"][self.hand_indices] = INSPIRE_OPEN_POSITION
    action["target_time"] = time.monotonic() + 0.5
    return action, 0.5  # action, wait_seconds
```

**Level 2 — fingers + slight pullback:**
```python
def reset_level_2(self, state):
    """Open Inspire fingers, pull arm back ~20% toward home."""
    action = self._get_hold_position_action(state["q"])
    # Open hand
    action["target_upper_body_pose"][self.hand_indices] = INSPIRE_OPEN_POSITION
    # Pull arm back slightly: 80% current + 20% home
    # WBC interpolation handles smooth motion automatically
    current_arm = state["q"][self.arm_indices]
    retract_target = 0.8 * current_arm + 0.2 * SAFE_HOME_ARM
    action["target_upper_body_pose"][self.arm_indices] = retract_target
    action["target_time"] = time.monotonic() + 1.0
    return action, 1.0
```

**Level 3 — full retract (safety only):**
```python
def reset_level_3(self, state):
    """Open Inspire fingers, retract arm fully to home."""
    action = self._get_hold_position_action(state["q"])
    action["target_upper_body_pose"][self.hand_indices] = INSPIRE_OPEN_POSITION
    action["target_upper_body_pose"][self.arm_indices] = SAFE_HOME_ARM
    action["target_time"] = time.monotonic() + 2.5
    return action, 2.5
```

#### Why this works

The GR00T model is **stateless** — every query is independent. It takes the current observation (arm pose + camera image) and plans from there. After a Level 1 reset:
- The arm is still near the object
- The hand is open
- The camera sees the current scene
- The model naturally plans a short re-grasp, not a full approach

This is the same reason the model works from any starting pose during normal operation. We're just exploiting that statelesness — letting the model recover from wherever it is rather than forcing it back to a known start.

#### Time comparison

```
Old approach (home retract):
  Open hand (0.5s) + retract to home (2.0s) + wait (0.5s) + approach (2-3s) = 5-6s per retry

New approach (smart reset):
  Level 1: 0.7s per retry (most grasps)
  Level 2: 1.5s per retry (object disturbed)
  Level 3: 3.0s per retry (safety, rare)

3 retries worst case: 2.1s (L1) vs 15-18s (old)
```

**Why NOT reset during transport:**

* During transport, Inspire fingers are closed around object

* Opening hand to "retry" means dropping the object — unrecoverable

* If object falls mid-transport, it's likely on the floor (out of reach)

* During transport, if per-finger `tau_est` drops (slip detected), the correct response is to TIGHTEN grip, not reset

* Grip tightening is a simple correction rule (no model needed):

  ```
  if transporting AND finger_tau_dropping:
      increase Inspire hand joint targets slightly (tighten)
  ```

  This goes in correction\_rules.py, not in the verifier

### 5.5 Camera configuration

Enable wrist cameras when launching the camera server on the robot:

```bash
python gr00t_wbc/control/sensor/composed_camera.py \
    --ego_view_camera oak \
    --left_wrist_camera oak \
    --right_wrist_camera oak \
    --fps 30 \
    --server True \
    --port 5556
```

The inference policy client already connects to this camera server. Wrist images will appear in the `images` dict alongside `ego_view`:

```python
# In format_observation or verify_grasp call:
images = camera_data["images"]
ego_view = images["ego_view"]           # already used
left_wrist = images["left_wrist"]       # now available
right_wrist = images["right_wrist"]     # now available
```

No code change in ComposedCameraSensor — it already supports wrist cameras via config. Just need to pass the config flags.

***

## Part 6: Keeping Robot Load Minimal

### What runs on robot vs server

| Component                    | Runs on                          | CPU cost                   | GPU cost                    | Network cost              |
| ---------------------------- | -------------------------------- | -------------------------- | --------------------------- | ------------------------- |
| Control loop (WBC)           | Robot                            | High (50Hz IK)             | None                        | None                      |
| Camera streaming             | Robot                            | Medium (JPEG encode)       | None                        | \~1MB/s per camera        |
| State publishing             | Robot                            | Low                        | None                        | \~10KB/s                  |
| **Inference policy client**  | **Robot**                        | **Low (format + publish)** | **None**                    | **\~30KB per query**      |
| **Verify grasp call**        | **Robot sends, server computes** | **\~0 (just ZMQ send)**    | **None on robot**           | **\~30KB once per grasp** |
| GR00T model inference        | Server                           | Low                        | High (3B model)             | \~30KB response           |
| **Grasp verifier inference** | **Server**                       | **Low**                    | **Negligible (2.5M model)** | **\~100 bytes response**  |

### What we explicitly avoid on robot

1. **No model loading on robot** — verifier runs on server
2. **No image preprocessing on robot** — server handles resize/normalize
3. **No PyTorch on robot side** — inference policy client uses only numpy
4. **No continuous verification** — only triggered once per grasp attempt
5. **No wrist camera processing on robot** — raw JPEG sent to server

### Network overhead per grasp verification

```
Robot → Server:
  wrist_image: ~30KB (JPEG compressed, already done by camera server)
  tau_est: 48 bytes (6 × float64)
  Total: ~30KB

Server → Robot:
  holding: 1 byte (bool)
  confidence: 8 bytes (float64)
  Total: ~10 bytes

Frequency: once per grasp attempt (not per control tick)
Latency: ~5ms round-trip on LAN
```

***

## Part 7: Full Implementation Steps

### Step 1: Enable wrist cameras (hardware setup, no code)

* Plug in wrist cameras (OAK-D or similar) to robot

* Launch camera server with wrist camera config flags

* Verify images stream: `python -c "from gr00t_wbc.control.sensor.composed_camera import ComposedCameraClientSensor; c = ComposedCameraClientSensor(...); print(c.read()['images'].keys())"`

* Should show: `dict_keys(['ego_view', 'left_wrist', 'right_wrist'])`

### Step 2: Collect dataset (1-2 hours of teleoperation)

* Run normal teleoperation sessions (operator doing pick tasks)

* Run `collect_grasp_data.py` in a 4th terminal

* Operator labels each grasp as success/failure

* Auto-negatives collected in background

* Target: 500+ labeled samples

### Step 3: Train model (\~15 minutes)

* Run `train_grasp_verifier.py` on GPU server

* Validate accuracy >93%, false negative rate <3%

* Export to TorchScript

### Step 4: Add server endpoint (\~30 lines in run\_gr00t\_server.py)

* Load TorchScript model at server startup

* Register `verify_grasp` endpoint

* Test: call endpoint manually with a test image

### Step 5: Add state machine to inference policy (\~100 lines in run\_gr00t\_inference\_policy.py)

* Add grasp detection (is\_grasp\_command)

* Add verify step after grasp chunk completes

* Add retract action

* Add retry counter (max 3)

* Add correction rules (gripper tightening during transport)

### Step 6: Hardening loop (run 100 trials)

* Run 100 pick attempts

* Log every verification result

* Categorize failures

* Retrain model with failure cases added to dataset

* Tune thresholds (GRASP\_CLOSE\_THRESHOLD, confidence threshold)

***

## Part 8: Files Summary

### New files to create

```
scripts/
├── verification/
│   └── grasp_verifier.py           # Model definition + inference wrapper
├── data_collection/
│   └── collect_grasp_data.py       # Dataset collection during teleop
├── training/
│   └── train_grasp_verifier.py     # Training script
└── correction_rules.py             # Gripper tightening + known failure fixes
```

### Existing files to modify

```
Isaac-GR00T/gr00t/eval/run_gr00t_server.py
  └─ Add: load grasp verifier, register "verify_grasp" endpoint

scripts/run_gr00t_inference_policy.py
  └─ Add: state machine, grasp detection, verify call, retract, retry
```

### Files that do NOT change

```
gr00t_wbc/          — entire directory unchanged
Isaac-GR00T/gr00t/model/     — model architecture unchanged
Isaac-GR00T/gr00t/policy/gr00t_policy.py  — unchanged
gr00t/policy/server_client.py  — unchanged (register_endpoint already supports custom endpoints)
```
