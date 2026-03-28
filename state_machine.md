# G1 Locomanipulation — Revised Implementation Plan

## Per-Skill DP3 + Vision-Verified Retry Architecture

**Target:** Unitree G1 (29-DOF) + Inspire DH56DFX Hands (NO tactile/force sensors)
**Cameras:** Intel RealSense D455 (head-mounted, RGB+depth) + 2× RGB wrist cameras (one per hand)
**Framework:** LeRobot (policy training/deployment) + WBC-AGILE (locomotion) + Custom State Machine (orchestration)
**Key Shift:** Per-skill DP3 policies replace the Bridge Think/Act hybrid. VLM (Cosmos-Reason2) does task decomposition only, not waypoint prediction.

---

## Why the Architecture Changed

The original plan (Bridge Think/Act) used a VLM to predict sparse 3D waypoints in camera frame, then an action expert conditioned on those waypoints to generate joint trajectories. This introduces compounding error sources:

1. VLM 2D anchor prediction errors
2. Camera-to-base frame transform errors (approximate extrinsics)
3. B-spline interpolation artifacts
4. Noise injection training to compensate for all of the above

Per-skill DP3 trained on 50-100 teleoperated demos per skill bypasses all of this. The policy learns directly from demonstrations — no VLM waypoints, no frame transforms, no B-spline. DP3 achieves 85-100% single-attempt success on real-world tasks with as few as 40 demos (DP3 paper). That baseline, combined with vision-verified retry, gets us to 95-98%.

**What we keep from the original:**
- APPF training paradigm (Phase 1 trajectory pre-training at large batch → Phase 2 point cloud fine-tuning)
- Noise injection (but applied to observations, not VLM waypoints)
- LeRobot plugin architecture
- Annotation pipeline (for APPF pre-training data)
- WBC-AGILE integration design

**What we drop:**
- VLM waypoint prediction pipeline (2D anchor → depth → 3D → B-spline)
- Camera-frame prediction (no longer needed — policy operates in joint space)
- Guidance pose conditioning in the action expert (DP3 is conditioned on point cloud + state only)
- VLM SFT for waypoint prediction

**What we add:**
- Per-skill DP3 policies with proper DP3 encoder (LayerNorm, not vanilla PointNet)
- Grasp primitive library with learned finger trajectories
- Vision-based grasp verification classifier (wrist cam)
- Motor current proxy for secondary grasp feedback
- Contact-phase impedance control for wipe/drawer skills (motor current as force proxy)
- VLM task decomposer (skill selection + parameterization, not waypoint generation)
- Perturbed retry system

**Conditional addition (Phase 3 — only if failure analysis warrants it):**
- Wrist-camera visual servoing correction loop (last 10-15cm of approach)
  Added only if 100-trial eval shows "last-cm alignment" as a dominant failure mode.
  DP3 with PCD already has 3D scene understanding — a second correction system may
  fight the policy rather than help it. Measure first, then decide.

---

## System Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      STATE MACHINE (Orchestrator)                        │
│   IDLE → PERCEIVE → PLAN → EXECUTE → VERIFY → SUCCESS / RETRY / FAIL   │
│   Pre/post condition checks │ Perturbed retry (3 attempts max)          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐                  ┌──────────────────────────────┐  │
│  │ VLM Task Planner  │  skill calls    │  Per-Skill DP3 Policies      │  │
│  │ (Cosmos-Reason2)  │───────────────▶│  (LeRobot plugin)            │  │
│  │                    │                 │                              │  │
│  │ Decomposes:        │  Point Cloud   │  Inputs:                     │  │
│  │ - instruction →    │───────────────▶│  - proprioceptive state (26) │  │
│  │   skill sequence   │                 │    (14 arm + 12 finger)     │  │
│  │ - object IDs       │                 │  - point cloud (Phase 2)     │  │
│  │ - grasp type       │                 │                              │  │
│  │                    │                 │                              │  │
│  │                    │                 │  Outputs:                    │  │
│  │                    │                 │  - 14 arm joint targets      │  │
│  │                    │                 │  - 1 grasp trigger flag      │  │
│  └──────────────────┘                  └──────────┬───────────────────┘  │
│                                                    │                      │
│  ┌──────────────────┐                              │ arm joints +         │
│  │ Grasp Primitive   │◀── grasp trigger ───────────┤ grasp trigger        │
│  │ Library           │         ┌────────────────┐  │                      │
│  │ (learned finger   │◀───────│ Grasp Type     │  │                      │
│  │  trajectories)    │        │ Classifier     │  │                      │
│  └───────┬──────────┘        │ (ResNet-18)    │  │                      │
│          │ finger cmds        └────────────────┘  │                      │
│          │                                         │                      │
│  ┌───────▼──────────┐              ┌──────────────▼───────────────────┐  │
│  │ Inspire Hand     │              │  WBC-AGILE Locomotion Policy     │  │
│  │ Controller       │              │  (RL-trained, sim-to-real)       │  │
│  │ + Motor Current  │              │  200Hz on Jetson Orin            │  │
│  │   Monitoring     │              │  Handles: balance, gait,         │  │
│  └──────────────────┘              │           arm perturbation       │  │
│                                     └──────────────┬─────────────────┘  │
│  ┌──────────────────┐              ┌──────────────▼───────────────────┐  │
│  │ Verification     │              │  G1 Hardware (29 DOF)            │  │
│  │ Pipeline         │              └──────────────────────────────────┘  │
│  │ - Grasp success  │                                                    │
│  │   classifier     │  ◀─── L/R wrist cam RGB ──┐                      │
│  │ - Motor current  │  ◀─── servo current ───────┤                      │
│  │ - Place verifier │  ◀─── RealSense RGB+D ─────┘                      │
│  │ - Slip detector  │                                                    │
│  └──────────────────┘                                                    │
│                                                                          │
│  ┌──────────────────┐  (wipe/drawer only)                               │
│  │ Contact-Phase    │  Activated when DP3 approach complete              │
│  │ Controller       │  Motor current as force proxy                     │
│  │ (impedance-like) │  Maintains target current during surface contact  │
│  │ 50-100Hz serial  │  Falls back to position control if current drops  │
│  └──────────────────┘                                                    │
│                                                                          │
│  ┌──────────────────┐  (PHASE 3 — conditional, only if failure          │
│  │ Wrist-Cam Visual  │   analysis shows last-cm alignment errors         │
│  │ Servoing Loop     │   as dominant failure mode)                       │
│  │ (ResNet-18, 30Hz) │  Adds (dx,dy,dz) correction to DP3 trajectory   │
│  │ FROZEN DP3 policy │  DP3 policy must be frozen when this is active   │
│  └──────────────────┘  to prevent two controllers fighting              │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Changes to Existing Code

### 1. Configuration (`configuration_waypoint_expert.py`)

**Current state:** Single config with `guidance_dim=7`, VLM waypoint conditioning assumed.

**Changes needed:**

```
REMOVE:
  - guidance_dim field (no longer conditioning on VLM waypoints)
  - noise_scale_position, noise_scale_orientation (waypoint noise — replaced with obs noise)
  - noise_scale_schedule
  - noise_injection (repurposed below)
  - training_phase logic that forces use_pointcloud=False

ADD:
  - skill_name: str field (e.g., "pick", "place", "wipe") — per-skill config
  - skill_id: int field (0-4) — used as conditioning token in APPF Phase 1 multi-skill training
  - obs_noise_position: float = 0.005  # sensor noise augmentation (m)
  - obs_noise_orientation: float = 0.02  # sensor noise augmentation (rad)
  - obs_noise_state: float = 0.01  # joint state noise augmentation (rad)
  - pcd_encoder_type: str = "dp3"  # "dp3" (with LayerNorm) or "pointnet" (legacy)
  - pcd_output_dim: int = 64  # DP3 paper: projection head compresses to exactly 64
  - use_colorless_pcd: bool = True  # DP3 paper: improves appearance robustness
  - sample_prediction: bool = True  # predict x0 instead of epsilon (DP3 paper)
  - crop_mode: str = "workspace"  # "workspace" (fixed table volume) or "object" (from detection bbox)
  - crop_center: list = [0.5, 0.0, 0.4]  # workspace crop center (x,y,z in base frame)
  - crop_radius: float = 0.3  # point cloud crop half-extent (m)

MODIFY:
  - action_dim: 26 → 15  (14 arm joints + 1 grasp trigger flag)
  - state_dim: 26 → 26   (KEEP at 26: 14 arm + 12 finger positions as observation)
      NOTE: Policy OBSERVES finger state (needed for grasp timing correlation)
      but only OUTPUTS arm joints + grasp trigger. Fingers are controlled
      by grasp primitives, triggered by the grasp flag.
      Head joint positions (3-DOF) are recorded as a SEPARATE observation key
      (observation.head_joints), NOT included in state_dim. They are only used
      by the processor's WorkspaceCropStep to compute camera→base frame transform.
      The DP3 policy does not need head joints — the PCD is already in base frame
      after the crop transform.
  - horizon: 16 → 4       (DP3 paper: shorter horizon = more reactive, H=4)
  - n_action_steps: 8 → 3 (DP3 paper: Nact=3)
  - hidden_dim: 256 → 256 (keep)
  - diffusion_steps: 100 → 100 (keep)
  - inference_steps: 10 → 10 (keep)

  Normalization mapping:
  - REMOVE guidance_pose normalization
  - ADD observation.pointcloud normalization (NONE — DP3 uses raw coords)

KEEP AS-IS:
  - @PreTrainedConfig.register_subclass("waypoint_expert")
  - get_optimizer_preset() (AdamW)
  - get_scheduler_preset() (CosineDecayWithWarmup)
  - validate_features() structure
  - training_phase ("phase1"/"phase2") and APPF batch size logic
  - delta_timestamps structure (adjust values for new horizon)
```

### 2. Model (`modeling_waypoint_expert.py`)

**Current state:** MLP encoders + simple PointNet + FFN-based ConditionalDiffusionModel.

**Changes needed:**

```
REMOVE:
  - GuidanceNoiseInjector class (entire class — no more waypoint noise)
  - guidance_encoder (nn.Sequential MLP for guidance_dim → hidden_dim)
  - All guidance_pose references in forward() and predict_action_chunk()
  - _quaternion_multiply helper (was only for guidance noise)

ADD:
  - DP3Encoder class (replaces PointNetEncoder):
      - MUST match the paper's exact architecture (Table VI shows minor changes cause huge swings)
      - Per-point MLP with LayerNorm and ReLU (NOT SiLU — paper uses ReLU explicitly)
      - Architecture (from paper appendix):
            self.mlp = nn.Sequential(
                nn.Linear(3, 64), nn.LayerNorm(64), nn.ReLU(),
                nn.Linear(64, 128), nn.LayerNorm(128), nn.ReLU(),
                nn.Linear(128, 256), nn.LayerNorm(256), nn.ReLU())
            self.projection = nn.Sequential(nn.Linear(256, 64), nn.LayerNorm(64))
      - 3-layer MLP (3→64→128→256), NOT 4 layers
      - Max-pool over points → global feature (256-dim)
      - Projection head compresses to exactly 64 dimensions (paper tested this)
      - NO BatchNorm (paper Table VI: BatchNorm is a "primary inhibitor")
      - NO T-Net (paper Table VI: T-Net hurts with fixed camera)
      - NO KNN local grouping (adds complexity; paper's simple encoder outperforms complex ones)

  - ObservationNoiseInjector class (replaces GuidanceNoiseInjector):
      - Adds Gaussian noise to state observations during training
      - Adds Gaussian noise to point cloud positions during training
      - Simulates real-world sensor noise, not VLM imprecision
      - position_noise_scale: 0.005m, state_noise_scale: 0.01rad

  - Replace ConditionalDiffusionModel internals:
      CURRENT: ResidualBlock = Linear → SiLU → Linear → skip connection (position-independent FFN)
      NEW:     TemporalResBlock = Conv1d(hidden, hidden, kernel=5, pad=2) → GroupNorm → SiLU
               → Conv1d(hidden, hidden, kernel=5, pad=2) → GroupNorm → skip connection
               + FiLM conditioning: scale, shift = Linear(condition_dim, 2*hidden).chunk(2)
                                    x = x * (1 + scale) + shift
      This gives the diffusion model temporal awareness across the action horizon.

  - sample_prediction option:
      CURRENT: predicts noise ε, loss = MSE(ε_pred, ε)
      NEW:     predicts clean sample x0, loss = MSE(x0_pred, x0_gt)
      DP3 paper shows sample prediction converges faster.

MODIFY:
  - forward():
      - Remove guidance encoding path
      - condition = state_feat (+ pcd_feat if Phase 2) (+ skill_id_embed if multi-skill Phase 1)
      - state_feat encodes full 26-dim state (14 arm + 12 finger positions)
      - Output is 15-dim action (14 arm joints + 1 grasp trigger flag)
      - Add ObservationNoiseInjector call on state and PCD during training

  - predict_action_chunk():
      - Remove guidance encoding path
      - condition = state_feat (+ pcd_feat if Phase 2) (+ skill_id_embed if present)

  - select_action():
      - Keep action chunking logic (deque-based queue)
      - Adjust for new n_action_steps=3
      - Extract grasp_trigger from action[:, -1] and pass to grasp primitive system
      - Return arm_joints=action[:, :14], grasp_trigger=action[:, 14]

  - PointNetEncoder → DP3Encoder:
      - Replace with exact paper architecture (3-layer MLP + projection to 64)
      - Use ReLU (not SiLU), LayerNorm (not BatchNorm)
      - Add use_colorless_pcd support (drop color channels if present)

  - Add SkillIDEmbedding (optional, for multi-skill Phase 1):
      - nn.Embedding(num_skills, 16)  # 16-dim learnable skill embedding
      - Concatenated to state_feat before conditioning
      - Only used during Phase 1 multi-skill training
      - Dropped during Phase 2 per-skill fine-tuning

KEEP AS-IS:
  - WaypointExpertPolicy class structure (config_class, name, reset())
  - DDPMScheduler / DDIMScheduler usage
  - Action chunking deque logic
  - SinusoidalPosEmb
  - Overall forward/select_action flow (just remove guidance path)
```

### 3. Processor (`processor_waypoint_expert.py`)

**Current state:** Renames `annotation.guidance_trajectory_base` → `observation.guidance_trajectory_base` → `guidance_pose`.

**Changes needed:**

```
REMOVE:
  - _waypoint_batch_to_transition() custom function (no more annotation.guidance renaming)
  - RenameObservationsProcessorStep for guidance_pose
  - All guidance-related processing

ADD:
  - PointCloudFromDepthProcessorStep:
      - Takes observation.images.depth (H, W, 1) + camera intrinsics
      - Deprojects to point cloud (H*W, 3)
      - Crop strategy (IMPORTANT — do NOT crop around current EE position):
          At trajectory start, EE is far from the target object. Cropping around
          EE would miss the object entirely. Instead use one of:
          (a) "workspace" mode: fixed crop volume around the table/workspace
              (e.g., centered at [0.5, 0.0, 0.4] in base frame, 0.3m radius)
              This is what DP3 paper uses — a workspace bounding box.
          (b) "object" mode: crop around detected object bbox center (from
              GroundingDINO). More precise but adds detection dependency.
          Default to "workspace" mode for reliability.
      - Downsamples via FPS to config.num_points (512 or 1024)
      - Drops color channels if config.use_colorless_pcd=True
      - Outputs observation.pointcloud (num_points, 3)
      - For Phase 1: this step is skipped (no PCD)
      - Depth comes from head-mounted RealSense D455

  - WorkspaceCropStep:
      - Crops point cloud to a fixed bounding box in robot base frame
      - Requires camera extrinsics (head camera → base frame via FK of head joints)
      - Uses observation.head_joints (3-DOF) to compute head FK → camera extrinsics
      - Bounding box defined by crop_center + crop_radius from config
      - Filters out floor, walls, robot body points

  - ActionSliceProcessorStep (CRITICAL — handles state/action dim asymmetry):
      The training data has 26-dim actions (14 arm + 12 finger) but we need
      15-dim actions (14 arm + 1 trigger flag). This step handles the conversion:
      
      During TRAINING (in batch_to_transition or equivalent):
        - Slice action[:, :14] for arm joints
        - Read pre-computed annotation.grasp_trigger from Parquet (written by run_annotation.py)
        - Concatenate: action_out = cat([arm_joints_14, grasp_trigger_1], dim=-1)  # (15,)
        NOTE: trigger labels are pre-computed in the annotation pipeline (not on-the-fly)
        because the pipeline already has keyframe indices + pre_shape_frames.
      
      During INFERENCE:
        - Policy outputs 15-dim: action[:, :14] → arm joints, action[:, 14] → trigger
        - No slicing needed (policy already outputs correct dim)
      
      This step ALSO affects:
        - NormalizerProcessorStep: needs separate stats for 26-dim state vs 15-dim action
          Use SEPARATE normalization configs, not a shared one.
        - UnnormalizerProcessorStep: only unnormalize the 15-dim action output
        - delta_timestamps: action timestamps now correspond to 15-dim, not 26-dim
        - LeRobot dataset schema: keep original 26-dim actions in Parquet, slice in processor

MODIFY:
  - make_waypoint_expert_pre_post_processors():
      - Remove guidance rename steps
      - Add ActionSliceProcessorStep (training only — slices 26→15 and computes trigger)
      - Add PointCloudFromDepthProcessorStep (Phase 2 only)
      - Add WorkspaceCropStep (Phase 2 only, uses observation.head_joints)
      - Split NormalizerProcessorStep into state_normalizer (26-dim) and action_normalizer (15-dim)
      - Keep AddBatchDimensionProcessorStep, DeviceProcessorStep

KEEP AS-IS:
  - Overall factory function signature
  - Integration with factory.py dynamic import

POSTPROCESSOR changes:
  - UnnormalizerProcessorStep: now handles 15-dim actions (not 26-dim)
  - Add TriggerExtractorStep: splits output into arm_joints (14) + trigger_flag (1)
  - DeviceProcessorStep: keep as-is
```

### 4. Annotation Pipeline (`training/annotation/`)

**Current state:** Computes FK → keyframes → waypoints → B-spline → dense guidance trajectory. Writes to Parquet.

**Changes needed:**

```
MOSTLY KEEP — the annotation pipeline is still useful for:
  - APPF Phase 1 pre-training (trajectory data with keyframe structure)
  - Grasp primitive labeling (gripper state annotations)
  - Visualization and data quality verification

MINOR CHANGES:
  - waypoint_annotator.py:
      ADD: grasp_type classification per keyframe (power, precision, lateral, etc.)
           based on finger joint configuration at grasp keyframes
      ADD: export per-skill episode lists (which episodes are "pick", "place", etc.)
      KEEP: everything else as-is

  - run_annotation.py:
      ADD: grasp_type column to Parquet output (annotation.grasp_type)
      ADD: annotation.grasp_trigger column — pre-computed per-frame trigger labels
           For each episode:
             1. Load keyframe annotations (already computed by waypoint_annotator)
             2. For frames within (keyframe_idx - pre_shape_frames) to keyframe_idx: 1.0
             3. All other frames: 0.0
           This is cleaner than computing trigger labels on-the-fly in
           ActionSliceProcessorStep — the annotation pipeline already has keyframe
           indices and pre_shape_frames, and Parquet storage is cheap.
      KEEP: everything else as-is

  - verify_annotations.py: KEEP AS-IS
  - visualize_annotations.py: KEEP AS-IS
  - patch_parquet_types.py: KEEP AS-IS
```

### 5. Factory Integration (`lerobot/src/lerobot/policies/factory.py`)

```
NO CHANGES NEEDED.
The dynamic plugin import fallback works for any registered policy.
The naming convention (configuration_* ↔ modeling_* ↔ processor_*) is preserved.
```

### 6. Package Init (`__init__.py`)

```
KEEP AS-IS. Exports remain:
  - WaypointExpertConfig
  - WaypointExpertPolicy
  - make_waypoint_expert_pre_post_processors
```

---

## New Components to Build

### Component 1: Grasp Primitive Library

**Location:** `state_machine/hands/grasp_primitives.py`

**Purpose:** Define and execute 5-8 grasp types with learned finger trajectories. Since we have NO force/tactile sensors, we cannot do reactive force-controlled closure. Instead, each primitive is a short learned finger trajectory (~0.5s, 15 frames at 30Hz) trained from teleoperated demos of that grasp type.

```python
# Grasp primitives:
#   1. power_grasp     — full hand wrap, large objects (mugs, bottles)
#   2. precision_pinch — thumb + index, small objects (pens, coins)
#   3. lateral_pinch   — thumb pad against index side, flat objects (cards, keys)
#   4. three_finger    — thumb + index + middle, medium objects (balls, fruit)
#   5. wide_wrap       — all fingers spread then close, irregularly shaped objects
#
# Each primitive stores:
#   - finger_trajectory: (T_grasp, 6) — 6 finger DOF trajectory (single hand)
#       NOTE: 6 DOF per hand. Initially right-hand only.
#       When bimanual support is added, extend to (T_grasp, 12) for both hands.
#   - pre_shape: (6,) — finger configuration before approach (single hand)
#   - pre_shape_frames: int — number of frames before contact to start pre-shaping (~5-10 at 30Hz)
#   - motor_current_threshold: float — expected current draw when grasping successfully
#
# Grasp trigger mechanism:
#   The DP3 policy outputs 15 dims: 14 arm joints + 1 grasp trigger flag.
#   The grasp trigger flag is a continuous value in [-1, 1] (via tanh).
#   When trigger > 0.5 for 3 consecutive timesteps (hysteresis), grasp primitive activates.
#
#   CRITICAL: trigger label timing in training data
#   The trigger label must fire BEFORE fingers start moving — the grasp primitive
#   needs pre_shape_frames to pre-shape the fingers before contact.
#   Label assignment:
#     - Find keyframe_index where finger state changes (from annotation pipeline)
#     - Set trigger=1.0 from (keyframe_index - pre_shape_frames) to keyframe_index
#     - Set trigger=0.0 everywhere else
#   This means the policy learns to fire the trigger early enough for the
#   primitive to pre-shape and execute by the time the arm reaches grasp pose.
#
#   Fallback: if learned trigger is unreliable (fires too early/late),
#   fall back to distance-based trigger using PCD-estimated object distance.
#
# Execution:
#   1. Pre-shape fingers when trigger first crosses 0.5 (3-frame hysteresis)
#   2. Continue DP3 arm trajectory during pre-shape (~0.3s overlap)
#   3. Execute learned finger closure trajectory open-loop (~0.5s, 15 frames at 30Hz)
#   4. Check motor current proxy for basic contact confirmation
#   5. Hand off to vision-based grasp verification (wrist cam)
```

**Training data:** Extract grasp segments from existing teleoperation demos. Cluster by finger configuration at contact. Each cluster becomes a primitive. Need ~20-30 examples per primitive type.

### Component 2: Vision-Based Grasp Verification

**Location:** `state_machine/perception/grasp_verifier.py`

**Purpose:** Binary classifier — is the robot holding an object? This is the **single most critical component** for achieving 95-98% reliability. Without force sensors, vision is the only way to verify grasps.

```
Architecture:
  - Input: left AND right wrist camera RGB crops, concatenated channel-wise → (224×224×6)
    Single 6-channel input is simpler and faster than two separate backbones.
    Modify ResNet-18 first conv: Conv2d(6, 64, 7, 2, 3) instead of Conv2d(3, 64, 7, 2, 3).
  - Backbone: ResNet-18 (pretrained ImageNet — init first conv by duplicating RGB weights)
  - Output: P(holding) ∈ [0, 1]
  - Training data: ~500 positive (holding object) + ~500 negative (empty hand / failed grasp)
  - Augmentation: color jitter, random crop, blur (simulate motion during transport)

Additional verifiers:
  - PlaceVerifier: RealSense RGB+D confirms object at target location + wrist cam confirms hand empty
  - SlipDetector: wrist cam re-check at 5Hz during transport
    (object still visible in expected region of hand?)
  - DepthVerifier: RealSense depth cross-check before action commit
    (does the PCD cluster match a graspable object? is approach collision-free?)

Motor current proxy (secondary signal):
  - Read Inspire servo motor current via SDK
  - Current spike during closure = contact
  - Current drop during transport = possible slip
  - NOT reliable enough as primary signal, but useful as confirmation
```

### Component 3: Wrist-Camera Visual Servoing Correction Loop (PHASE 3 — CONDITIONAL)

**Location:** `state_machine/perception/wrist_servoing.py`
**Status:** NOT built in initial implementation. Added only if 100-trial failure analysis (weeks 16-17) shows "last-cm alignment errors" as a dominant failure mode.

**Why deferred:** DP3 with point cloud input (Phase 2) already has 3D scene understanding.
The policy sees the object in the PCD and generates joint trajectories that account for
object position. Adding a separate correction loop risks two systems fighting over the
same joints. Example: DP3 sees the mug is 2cm left in the PCD and curves left. Wrist
servoing also sees 2cm left and adds +2cm dx correction. Result: 2cm overshoot.

The prior work showing 10-15% improvement from eye-in-hand correction used policies
WITHOUT PCD conditioning — they were correcting a blind trajectory. DP3 is not blind.

**If failure analysis warrants it:**

```
Architecture:
  - Input: active wrist camera RGB crop (224×224) — whichever hand is approaching
  - Backbone: ResNet-18 or MobileNetV3 (must fit in <15ms on Orin)
  - Output: (dx, dy, dz) correction in EE frame, clipped to ±2cm
  - CRITICAL: DP3 policy weights MUST be frozen when servoing is active.
    Do not allow two learned systems to fight. The servoing correction is
    additive on top of a fixed DP3 base trajectory.
  - Activation: only when EE is within 15cm of target, fades in linearly
  - Frequency: 30Hz on Orin (NOT offboard — latency requirement <15ms)
  
Training data:
  - Extract from teleop demos: pair wrist images with expert's corrective motion
  - Augment with synthetic offsets: shift wrist images ±20px
  - Need ~20-30 demos with deliberately varied initial approach offset
```

### Component 4: Contact-Phase Controller (Wipe / Drawer Skills)

**Location:** `state_machine/skills/contact_controller.py`

**Purpose:** DP3 is designed for free-space manipulation (reach, pick, place). It has NOT been tested on sustained-contact tasks like wiping or drawer opening. For these skills, DP3 handles the approach phase, then a separate contact-phase controller takes over once contact is detected.

```
Design:
  The DP3 policy runs the approach — moving the arm to the surface/handle.
  Once contact is detected (motor current spike > threshold), control switches
  to a simple impedance-like controller using motor current as a force proxy.

  For WIPE:
    - DP3 approaches the surface
    - Contact detected via motor current spike in wrist/finger servos
    - Switch to contact controller:
        - Normal axis: maintain target motor current (≈ target force)
          If current drops below threshold → push harder (lower Z)
          If current exceeds threshold → back off (raise Z)
        - Lateral axes: follow a pre-programmed wipe pattern (zigzag, circular)
          loaded from the skill config, parameterized by surface bbox
        - Lateral motion speed: 5-10cm/s (slow enough for 50-100Hz feedback)
    - Contact controller runs at 50-100Hz via serial polling of Inspire servos
      NOTE: Inspire DH56DFX communicates over UART/CAN. Reading motor current
      from all 6 finger servos at 500Hz simultaneously is not achievable with
      serial polling. Typical servo current readback is 50-100Hz. This limits
      impedance bandwidth but is sufficient for slow wipe/drawer motions.
    - DP3 policy is paused during contact phase

  For OPEN_DRAWER:
    - DP3 approaches the handle
    - Grasp primitive activates (handle grasp)
    - Switch to contact controller:
        - Pull direction: constant velocity pull along drawer axis
        - Compliance: if motor current spikes (drawer stuck), reduce pull force
        - Termination: drawer position > target distance OR motor current drops (handle lost)
    - The pull direction is estimated from the initial handle detection bbox orientation

  Motor current → force mapping:
    - Inspire servos report raw current in mA
    - Approximate linear mapping: force ≈ k * (current - idle_current)
    - Calibrate k per-finger by pressing against a known surface with a scale
    - This is crude (~±30% accuracy) but sufficient for impedance-like control
    - Not reliable enough for force feedback during free-space grasping,
      but fine for maintaining surface contact during wipe/drawer

  Fallback:
    - If contact is lost (current drops below threshold for >0.5s), abort and retry
    - If motor current exceeds safety limit, emergency stop
```

### Component 5: State Machine with Retry

**Location:** `state_machine/core/`

```
state_machine/core/
├── engine.py              # StateMachineEngine: IDLE → PERCEIVE → PLAN → EXECUTE → VERIFY
├── skill_base.py          # Skill base class with pre/post conditions
├── retry.py               # Perturbed retry logic (angle ±10-15°, height ±1-2cm)
└── monitor.py             # Runtime monitoring at 5Hz (slip detection, tracking)
```

**State machine flow:**
```
IDLE
  │
  ▼
PERCEIVE ─── get RGB + depth, run object detection, build scene graph
  │
  ▼
PLAN ─────── VLM decomposes instruction → skill sequence with params
  │           (which skill, which object, which grasp type)
  ▼
EXECUTE ──── run per-skill DP3 policy + grasp primitive
  │
  ▼
VERIFY ───── vision-based verification (grasp classifier, place verifier)
  │
  ├── success → next skill or SUCCESS
  │
  ├── failure + retries_left > 0 → RETRY (perturb approach, retry)
  │     perturbation: angle ±10-15°, height ±1-2cm, alternate grasp primitive
  │
  └── failure + retries_left == 0 → safe retreat → FAIL
```

**Retry perturbation strategy:**
- Attempt 1: original plan
- Attempt 2: rotate approach angle +12°, raise 1.5cm
- Attempt 3: rotate approach angle -12°, try alternate grasp primitive
- Each retry: re-perceive (don't assume scene is unchanged)

### Component 6: VLM Task Decomposer

**Location:** `state_machine/planning/task_decomposer.py`

**Purpose:** VLM does task decomposition ONLY — not waypoint prediction. Given an instruction and RGB image, output a sequence of skill calls with parameters.

```
Input:
  - instruction: "Pick up the red mug and put it on the shelf"
  - RGB image from head camera
  - scene graph (detected objects with bounding boxes)

Output:
  [
    SkillCall("pick", {
      "object": "red_mug",
      "bbox": [120, 80, 200, 240],
      "grasp_type": "power",       # from grasp classifier
      "approach": "top"
    }),
    SkillCall("place", {
      "target": "shelf_surface",
      "bbox": [300, 100, 450, 200],
      "placement": "upright"
    })
  ]

VLM model: Nvidia Cosmos-Reason2-2B (default) or Cosmos-Reason2-8B (if needed)
  Built on Qwen3-VL, post-trained with physical AI reasoning data (EgoExo4D,
  Language Table, IntPhys, CLEVRER). Purpose-built for embodied task planning.
  Structured <think>/<answer> output format. 24GB VRAM min (2B, BF16).
  Repo: https://github.com/nvidia-cosmos/cosmos-reason2
  Model: https://huggingface.co/nvidia/Cosmos-Reason2-2B
This is MUCH simpler than waypoint prediction — just structured task decomposition.
No SFT needed — Cosmos-Reason2 is already trained on embodied reasoning data.
```

### Component 7: Perception Pipeline

**Location:** `state_machine/perception/`

```
state_machine/perception/
├── realsense_interface.py # RealSense D455 driver: aligned RGB+depth at 30Hz
├── point_cloud.py         # RealSense depth → point cloud → crop → FPS downsample
├── object_detector.py     # Object detection for scene graph (GroundingDINO or OWLv2)
├── grasp_verifier.py      # Vision-based grasp success classifier (6-ch dual wrist cam input)
└── grasp_classifier.py    # Object → grasp type classifier (ResNet-18)
```

**Camera roles (phase-dependent):**
- **RealSense (head):** Scene understanding — object detection, point cloud for DP3, place verification, depth-based collision checking. Provides aligned RGB+depth at 640×480. Dominates during all phases.
- **Left wrist cam:** Grasp verification (left hand), slip detection during transport. Phase 3: visual servoing during final approach.
- **Right wrist cam:** Grasp verification (right hand), slip detection during transport. Phase 3: visual servoing during final approach.
- **Phase-dependent priority:**
    - Approach phase: head RealSense drives DP3, wrist cams recording but passive
    - Grasp execution (fingers closing): wrist cam is primary verification sensor (head can't see through hand)
    - Carry/transport: head camera dominates, wrist cam confirms object retention at 5Hz
    - Contact phase (wipe/drawer): wrist cams not used (motor current is primary feedback)

**Perception verification (from RS4L):**
Before committing to action, cross-check:
1. Does the detected object have a reasonable point cloud cluster? (RealSense depth sanity — not a depth artifact)
2. Is the object within reachable workspace? (IK check)
3. Does the chosen grasp type match the object geometry?
4. Is the approach path collision-free? (depth-based check using RealSense)
If any check fails → re-perceive from a different head angle (G1 has 3-DOF head).

---

## Revised Data Collection Plan

### Current Data
- `shivubind/g1_pick_place`: 156 episodes, state (26,), action (26,), RGB only (no depth in this dataset)
- Cameras available NOW: RealSense D455 (head, RGB+depth) + 2× wrist RGB cameras
- Existing dataset usable for Phase 1 APPF pre-training (arm trajectory learning)
- **LIMITATION:** Existing 156 episodes do NOT record head joint positions or depth.
  Approximate fixed extrinsics were used. Usable for Phase 1 only.
- New data collection will include depth from RealSense — no monocular estimation needed

### New Data Collection Observation Schema

**ALL new episodes MUST record:**
```
observation.state                    # (26,) — 14 arm + 12 finger positions
observation.head_joints              # (3,) — G1 head pan/tilt/roll (REQUIRED for PCD crop)
observation.images.head_rgb          # 640×480×3, head RealSense
observation.images.head_depth        # 640×480×1, head RealSense (16-bit mm)
observation.images.wrist_left_rgb    # 320×240×3, left wrist cam
observation.images.wrist_right_rgb   # 320×240×3, right wrist cam
observation.motor_currents           # (12,) — Inspire servo currents (for contact controller calibration)
action                               # (26,) — 14 arm + 12 finger (processor slices to 15 for training)
```

The head_joints are CRITICAL — without them the camera→base frame transform is wrong
and the workspace PCD crop will be misaligned. The existing dataset cannot be used for
Phase 2 PCD fine-tuning because it lacks head_joints.

### New Data Collection (Priority Order)

| Skill | Episodes Needed | Camera Setup | Notes |
|-------|----------------|--------------|-------|
| pick_power | 80-100 | Full schema above | Large objects: mugs, bottles, bowls |
| pick_precision | 60-80 | Full schema above | Small objects: pens, keys, coins |
| place | 60-80 | Full schema above | Varied surfaces: table, shelf, tray |
| wipe | 40-50 | Full schema above | Table wiping, different patterns |
| open_drawer | 40-50 | Full schema above | Pull and push drawers |

**Episode counts are higher than the DP3 paper (40 demos) because:**
The DP3 paper reports 85-100% on a fixed-base WidowX arm with a fixed overhead camera.
The G1 setup differs in three ways that reduce expected transfer:
1. Head-mounted RealSense moves with the head → PCD viewpoint varies across demos
2. G1 humanoid has whole-body dynamics → arm positioning accuracy is lower than a fixed-base arm
3. Inspire DH56DFX has more DOF and mechanical backlash than a simple parallel-jaw gripper
Conservative initial estimate: 75-85% single-attempt pick (not 90%). This still works with
3 retries: 1 - (1-0.75)^3 = 98.4%. But plan for needing 80-100 demos per skill, not 50-60.

**Camera setup is already complete:** Head-mounted RealSense D455 provides RGB+depth. Two wrist RGB cameras provide close-up views for grasp verification. No monocular depth estimation needed.

### Grasp Verification Training Data

| Class | Examples Needed | Source |
|-------|----------------|--------|
| holding_object | 500 | Extract from teleop demos (post-grasp frames) |
| empty_hand | 300 | Extract from teleop demos (pre-grasp frames) |
| failed_grasp | 200 | Collect intentional failures (object slips, misalignment) |

Collect the failure examples intentionally — they're the most important training data and don't occur naturally in successful teleop demos.

**CRITICAL: Dedicated failure data collection session (block 1 full day in Week 3-4)**
Failed grasps are the most valuable training data for the verification classifier, and they
don't happen during normal teleoperation. Schedule a dedicated session where you deliberately:
- Command grasps offset by 2-3cm from the object center
- Use wrong grasp type for the object (precision pinch on a large bottle)
- Approach from suboptimal angles (too steep, too shallow)
- Grasp on object edges where contact is marginal
- Close fingers on thin air (complete miss)
- Grasp and then shake to induce slips during transport
Record both wrist cameras during all of these. The negative examples should cover
the full spectrum of failure modes, not just "hand is empty."

---

## Training Strategy

### Per-Skill DP3 Training (using existing LeRobot plugin)

**Phase 1: Trajectory Pre-training (APPF)**
```bash
# Train on existing g1_pick_place data (no PCD)
# This learns the arm motion manifold across all skills
# IMPORTANT: skill_id token is added to conditioning so the model can
# disambiguate skill types during multi-skill pre-training.
# Without it, pick and place trajectories are mixed with no identity signal,
# which can confuse the model when they have different motion profiles.
python -m lerobot.scripts.lerobot_train \
    --policy.type=waypoint_expert \
    --policy.training_phase=phase1 \
    --policy.use_pointcloud=false \
    --policy.skill_name=all \
    --policy.use_skill_id=true \
    --dataset.repo_id=shivubind/g1_pick_place \
    --training.batch_size=4096 \
    --training.lr=1e-4 \
    --training.num_steps=50000
```

NOTE on Phase 1 multi-skill training: The skill_id is a learnable embedding
(dim=16) concatenated to the state features. This gives the model a way to
distinguish "this is a pick trajectory" from "this is a place trajectory"
without needing separate models. Alternative: train Phase 1 per-skill too
(cleaner signal but smaller dataset per model). Start with multi-skill + ID,
fall back to per-skill if convergence is poor.

**Phase 2: Per-Skill Fine-tuning with Point Cloud**
```bash
# Fine-tune per skill with PCD (need new data with depth)
python -m lerobot.scripts.lerobot_train \
    --policy.type=waypoint_expert \
    --policy.training_phase=phase2 \
    --policy.use_pointcloud=true \
    --policy.skill_name=pick_power \
    --policy.pcd_encoder_type=dp3 \
    --policy.use_colorless_pcd=true \
    --policy.sample_prediction=true \
    --resume_from=outputs/phase1/checkpoints/best.pt \
    --dataset.repo_id=merai/g1_pick_power \
    --training.batch_size=256 \
    --training.lr=3e-5 \
    --training.num_steps=30000
```

### Grasp Verification Classifier Training
```bash
# Separate from LeRobot — standard PyTorch classification
python train_grasp_verifier.py \
    --backbone resnet18 \
    --pretrained imagenet \
    --data_dir data/grasp_verification/ \
    --epochs 50 \
    --lr 1e-4 \
    --augmentation strong
```

---

## Verification Stack (Without Force Sensors)

| Check | Method | Frequency | Accuracy Target |
|-------|--------|-----------|----------------|
| Grasp success | Wrist cam classifier (6-ch ResNet-18) + motor current | After grasp | >95% |
| Object slip during transport | Wrist cam re-check | 5 Hz | >90% |
| Place success | RealSense RGB+D (object at target) + wrist cam (hand empty) | After place | >95% |
| Contact detection (wipe/drawer) | Motor current spike above threshold | During execution | >85% |
| Contact maintenance (wipe/drawer) | Motor current stays in target band | 50-100 Hz | >90% |
| Arm tracking | Joint position error vs commanded | 50 Hz | Threshold-based |
| Balance monitoring | IMU orientation deviation | 200 Hz (WBC) | Threshold-based |
| **Final approach correction** | **Wrist cam visual servoing (Phase 3, conditional)** | **30 Hz** | **<1cm residual** |

**Motor current proxy details:**
- Inspire DH56DFX servos report current draw via SDK
- Current > threshold during closure ≈ "something is being gripped"
- Current drop during transport ≈ "object may have slipped"
- Noisy and uncalibrated — ~70% accuracy as standalone signal
- Combined with vision classifier: effective accuracy >95%

---

## Reliability Math

### Single-Attempt Success Rates (Conservative Estimates for Humanoid)

These estimates are MORE conservative than the DP3 paper numbers because
of the humanoid-specific challenges (see data collection section above).

| Component | Success Rate | Notes |
|-----------|-------------|-------|
| DP3 arm trajectory | 80% | 80-100 demos; humanoid arm accuracy < fixed-base arm |
| Grasp primitive execution | 80% | Learned finger trajectory, no force feedback, Inspire backlash |
| Grasp verification accuracy | 95% | ResNet-18 on dual wrist cam (6-channel input) |
| **Pick (combined)** | **~64%** | 0.80 × 0.80 |
| Place | ~82% | Arm trajectory × release timing |

### With Vision-Verified Retry (3 attempts)

```
P(pick_success_3_tries) = 1 - (1 - 0.64)^3 = 95.3%
P(place_success_3_tries) = 1 - (1 - 0.82)^3 = 99.4%
P(pick_and_place) = 0.953 × 0.994 = 94.7%
```

This is below the 98% target. To close the gap, ordered by expected impact:
1. More demos (100+ per skill) → arm trajectory 80% → 85%, pick combined 68%
   → 3-retry: 96.7%
2. Phase 3: Add wrist-cam visual servoing IF failure analysis confirms
   last-cm alignment as dominant mode → grasp primitive 80% → 88%
   → pick combined 75% → 3-retry: 98.4%
3. Grasp primitive tuning from failure analysis → 80% → 85%
   → pick combined 72% → 3-retry: 97.8%

The 98% target is achievable but requires the full eval→fix loop (weeks 14-20),
not just the initial architecture. Plan for iterating.

This math ONLY works if:
1. Verification classifier correctly identifies failures (95%+ accuracy)
2. Retry uses perturbation (not same trajectory replay)
3. Scene re-perception happens between retries

If verification accuracy drops to 85%, effective retry success drops to ~93% — still acceptable but leaves less margin.

---

## 20-Week Implementation Timeline

### Weeks 1-4: Foundation + Grasp System

**Week 1-2: Modify existing policy plugin + PCD quality validation**
- [ ] Update `configuration_waypoint_expert.py` (remove guidance, add DP3 params)
- [ ] Update `modeling_waypoint_expert.py` (DP3Encoder, TemporalResBlock + FiLM, remove guidance path)
- [ ] Update `processor_waypoint_expert.py` (remove guidance processing, add PCD-from-depth)
- [ ] Verify Phase 1 training still converges on `shivubind/g1_pick_place`
- [ ] **PCD Quality Validation with RealSense D455 (DECISION GATE):**
    - Capture D455 depth frames of target objects at manipulation distance (0.5-1m)
    - Apply RealSense SDK filters: temporal (α=0.4), spatial (σ=0.5), hole-filling
    - Convert to PCD, apply workspace crop + FPS downsample to 512 points
    - Visually inspect: are object shapes distinguishable? Are edges clean enough?
    - Compare against DP3 paper's example PCDs (L515 quality)
    - Test on challenging objects: reflective mug, thin pen, transparent bottle
    - Measure depth noise σ at 0.5m and 1.0m → set obs_noise_position accordingly
    - **GO/NO-GO decision:**
      - GO: D455 PCD quality sufficient → continue with DP3 architecture
      - NO-GO: PCD too noisy → pivot to original Diffusion Policy (RGB-only)
        with fixed head pose during manipulation. Code change is small:
        replace DP3Encoder with ResNet18Encoder, remove PCD processor steps.
        Diffusion backbone (temporal Conv1d + FiLM) stays identical.

**Week 3-4: Grasp primitives + verification**
- [ ] Build `state_machine/hands/grasp_primitives.py` — 5 grasp types with learned finger trajectories
- [ ] Extract grasp segments from existing 156 episodes, cluster into primitives
- [ ] Build `state_machine/perception/grasp_verifier.py` — ResNet-18 binary classifier (6-channel L+R wrist)
- [ ] **Dedicated failure data collection day (CRITICAL):**
    - Deliberately command offset grasps (±2-3cm), wrong grasp types, edge grasps
    - Record wrist cam + motor current for all failures
    - Collect ~200 failed grasp examples across all failure modes
- [ ] Collect ~1000 wrist cam images (holding/not_holding) from existing data + intentional failures
- [ ] Implement motor current reading via Inspire SDK
- [ ] Test grasp verification accuracy on held-out data (target: >95%)

### Weeks 5-8: Per-Skill DP3 Training

**Week 5-6: Data collection**
- [ ] Calibrate RealSense D455 extrinsics (head-mounted) and wrist cam extrinsics
- [ ] **Verify head joint recording (3-DOF) works in teleop pipeline — this is REQUIRED for PCD crop**
- [ ] Collect 80-100 pick_power episodes with full observation schema (depth + head joints + wrist cams + motor currents)
- [ ] Collect 60-80 pick_precision episodes with full observation schema
- [ ] Collect 60-80 place episodes with full observation schema
- [ ] Run annotation pipeline on new data (waypoint_annotator.py — still useful for APPF + trigger labels)

**Week 7-8: Training**
- [ ] Phase 1: multi-skill trajectory pre-training on combined dataset (APPF)
- [ ] Phase 2: per-skill fine-tuning with point cloud (pick_power, pick_precision, place)
- [ ] Ablation: DP3 encoder (LayerNorm) vs vanilla PointNet — verify improvement
- [ ] Ablation: sample prediction vs epsilon prediction — verify convergence speed
- [ ] Train grasp type classifier (object crop → grasp type)

### Weeks 9-10: WBC-AGILE Integration

- [ ] Deploy WBC-AGILE locomotion policy on Orin
- [ ] Test arm overlay on WBC (upper-body joint targets at 30-50Hz)
- [ ] Test walk-to-reach → stand → manipulate pipeline
- [ ] Verify balance during arm extension (with and without payload)
- [ ] If balance issues: retrain WBC with arm perturbation domain randomization in Isaac Lab

### Weeks 11-13: State Machine + Retry System

**Week 11: State machine core + contact controller**
- [ ] Build `state_machine/core/engine.py` — state machine with pre/post condition checks
- [ ] Build `state_machine/core/retry.py` — perturbed retry (angle ±10-15°, height ±1-2cm)
- [ ] Build `state_machine/core/monitor.py` — runtime monitoring at 5Hz
- [ ] Build `state_machine/skills/contact_controller.py` — impedance-like controller for wipe/drawer
- [ ] Calibrate motor current → force mapping per finger (press against known surface with scale)
- [ ] Test contact controller standalone: can it maintain surface contact during a wipe pattern?

**Week 12: Perception pipeline**
- [ ] Build `state_machine/perception/realsense_interface.py` — RealSense D455 aligned RGB+depth streaming
- [ ] Build `state_machine/perception/point_cloud.py` — RealSense depth → PCD → crop → FPS downsample
- [ ] Build `state_machine/perception/object_detector.py` — GroundingDINO for scene graph
- [ ] Perception verification loop (RS4L): cross-check PCD cluster, IK reachability, grasp type match

**Week 13: VLM task decomposer + end-to-end integration**
- [ ] Build `state_machine/planning/task_decomposer.py` — Cosmos-Reason2 for skill decomposition
- [ ] Wire everything together: VLM → skill sequence → DP3 → grasp → verify → retry
- [ ] End-to-end test: "pick up the mug" from instruction to grasp verification
- [ ] End-to-end test: "wipe the table" using DP3 approach + contact controller

### Weeks 14-20: Evaluation + Hardening

**Week 14-15: Build MuJoCo eval environment**
- [ ] Tabletop scene with graspable objects (5-8 object types)
- [ ] Automated trial runner: 100 trials per skill
- [ ] Failure categorization: log failure type, step, and observation at failure

**Week 16-17: First 100-trial evaluation round**
- [ ] Pick (stationary): target 85%+ single attempt
- [ ] Place: target 88%+ single attempt
- [ ] Pick + place (combined with retry): target 95%+
- [ ] Categorize every failure into buckets

**Week 18-19: Targeted fixes per failure category**
- [ ] For each failure bucket, engineer a specific fix:
    - Arm trajectory failures → collect more demos for that scenario
    - Grasp failures → add/tune grasp primitives, improve pre-shape
    - **Last-cm alignment errors (if dominant) → Phase 3: add wrist-cam visual servoing**
        - Collect ~30 offset-approach demos for servoing training data
        - Train ResNet-18/MobileNetV3 correction network
        - Integrate with FROZEN DP3 policy (do not retrain DP3)
        - Re-evaluate: expect pick combined to improve from ~64% to ~75%
    - Verification false negatives → augment classifier training data
    - Perception errors → tune depth/PCD pipeline, adjust crop radius
    - Balance issues → WBC domain randomization
- [ ] Re-run 100-trial evaluation after each fix

**Week 20: Final evaluation + sim-to-real prep**
- [ ] Final 100-trial evaluation per skill
- [ ] Document per-failure fixes and remaining failure modes
- [ ] Prepare real hardware deployment checklist
- [ ] Noise injection ablation on real hardware (if available)

---

## Sim-to-Real Strategy

This plan trains primarily on **real teleoperated data** (not sim). The MuJoCo eval environment
(weeks 14-15) is for automated evaluation loops, not for generating training data.

**Why real-only for now:**
- DP3 achieves 85-100% with 40-60 real demos (paper results) — sim augmentation not needed to hit targets
- Sim-to-real PCD transfer is an open research problem (sim depth is perfect, real depth is noisy)
- G1 + Inspire sim fidelity in MuJoCo/Isaac is unverified for finger contact dynamics
- Engineering time is better spent on more real demos than on sim pipeline

**When to add sim (future work):**
- If real data collection becomes the bottleneck (>100 demos per skill needed)
- If PCD diversity is insufficient (limited object set in lab)
- Approach: use Isaac Lab to generate point cloud training data with domain randomization
  (random objects, textures, lighting). Train DP3 Phase 2 on mixed real+sim PCD.
  Keep Phase 1 real-only (trajectory dynamics should come from real robot).
- Requires: verified G1+Inspire URDF in Isaac Lab, realistic depth sensor noise model,
  validated contact dynamics for Inspire fingers

---

## File Structure (Final)

```
state_machine/
├── CLAUDE.md                                # Updated with new architecture
├── state_machine.md                         # THIS DOCUMENT
│
├── lerobot_policy_waypoint_expert/          # LeRobot policy plugin (MODIFIED)
│   ├── pyproject.toml
│   └── src/lerobot_policy_waypoint_expert/
│       ├── __init__.py
│       ├── configuration_waypoint_expert.py # MODIFIED: remove guidance, add DP3 params
│       ├── modeling_waypoint_expert.py      # MODIFIED: DP3Encoder, TemporalResBlock+FiLM
│       └── processor_waypoint_expert.py     # MODIFIED: remove guidance, add PCD-from-depth
│
├── training/                                # Data pipeline (MOSTLY KEPT)
│   ├── annotation/
│   │   ├── waypoint_annotator.py            # MINOR CHANGES: add grasp_type classification
│   │   ├── run_annotation.py               # MINOR CHANGES: add grasp_type column
│   │   ├── verify_annotations.py           # KEPT AS-IS
│   │   ├── visualize_annotations.py        # KEPT AS-IS
│   │   └── patch_parquet_types.py          # KEPT AS-IS
│   ├── train_grasp_verifier.py             # NEW: ResNet-18 grasp classifier training (6-ch input)
│   ├── train_grasp_type_classifier.py      # NEW: object → grasp type classifier
│   └── upload_to_hf.py                     # KEPT AS-IS
│
├── state_machine/                           # NEW: orchestration layer
│   ├── core/
│   │   ├── engine.py                       # State machine: IDLE → PERCEIVE → PLAN → ...
│   │   ├── skill_base.py                   # Skill base class with pre/post conditions
│   │   ├── retry.py                        # Perturbed retry logic
│   │   └── monitor.py                      # Runtime monitoring at 5Hz
│   ├── skills/
│   │   ├── pick_skill.py                   # Per-skill DP3 execution + grasp primitive
│   │   ├── place_skill.py
│   │   ├── wipe_skill.py                   # DP3 approach + contact_controller for contact phase
│   │   ├── open_drawer_skill.py            # DP3 approach + contact_controller for pull phase
│   │   └── contact_controller.py           # NEW: impedance-like controller for sustained contact
│   ├── planning/
│   │   └── task_decomposer.py              # VLM task decomposition (not waypoint prediction)
│   ├── perception/
│   │   ├── realsense_interface.py          # RealSense D455: aligned RGB+depth at 30Hz
│   │   ├── point_cloud.py                  # RealSense depth → PCD → workspace crop → FPS
│   │   ├── object_detector.py              # GroundingDINO for scene graph
│   │   ├── grasp_verifier.py               # Vision-based grasp success (6-ch dual wrist cam)
│   │   └── grasp_classifier.py             # Object → grasp type classifier
│   ├── hands/
│   │   ├── grasp_primitives.py             # 5 grasp types with learned finger trajectories
│   │   └── inspire_controller.py           # Inspire SDK wrapper + motor current monitoring
│   └── locomotion/
│       └── wbc_interface.py                # WBC-AGILE interface (same as original design)
│
├── lerobot/                                 # LeRobot fork (KEPT AS-IS)
│   └── src/lerobot/policies/factory.py     # Plugin discovery — no changes needed
│
├── g1_description/                          # Robot URDF/meshes (KEPT)
│   └── inspire_hand/config.yaml
│
├── configs/                                 # NEW
│   ├── g1_inspire.yaml                     # Hardware config (camera intrinsics, joint limits)
│   ├── skills.yaml                         # Per-skill config (which policy, which grasp type)
│   ├── training_phase1.yaml                # APPF Phase 1 training config
│   └── training_phase2_pick.yaml           # Per-skill Phase 2 training config
│
└── evaluation/                              # NEW
    ├── eval_runner.py                       # Automated 100-trial evaluation
    ├── failure_categorizer.py               # Classify failures into buckets
    └── configs/
        ├── eval_pick.yaml
        └── eval_place.yaml
```

---

## Summary: Original vs Revised

| Aspect | Original (Bridge Hybrid) | Revised (Per-Skill DP3) |
|--------|-------------------------|------------------------|
| **Action policy** | Single waypoint-conditioned expert | Per-skill DP3 (no waypoint conditioning) |
| **VLM role** | Predict 3D waypoints in camera frame | Task decomposition only (skill + params) |
| **Point cloud encoder** | Simple 3-layer PointNet | DP3 encoder with LayerNorm (exact paper arch, 64-dim output) |
| **Diffusion network** | FFN blocks (position-independent) | 1D temporal conv + FiLM conditioning |
| **Prediction type** | Epsilon prediction | Sample prediction (DP3 paper: faster convergence) |
| **Action space** | 26-dim (arm + fingers) | 15-dim (14 arm + 1 grasp trigger flag) |
| **Observation space** | 26-dim state | 26-dim state (14 arm + 12 finger) + 3-dim head joints (separate key) |
| **PCD crop strategy** | EE-centered | Workspace-centered (fixed table volume) |
| **Finger control** | End-to-end learned (26-dim action) | Grasp primitives triggered by learned flag (with pre-shape timing) |
| **Wrist cameras** | Not used for control | Grasp verification + slip detection (Phase 3: visual servoing if needed) |
| **Contact-rich skills** | Same as free-space | Separate contact-phase controller (motor current at 50-100Hz) |
| **Grasp verification** | Force sensors (NOT AVAILABLE) | 6-channel wrist cam classifier + motor current proxy |
| **Failure recovery** | Implied | Explicit: perturbed retry, 3 attempts |
| **Training data needed** | Combined multi-skill | Per-skill (80-100 demos each) + failure examples |
| **APPF training** | Kept | Kept (Phase 1 with skill_id token for multi-skill) |
| **Noise injection** | On VLM waypoints (simulates VLM error) | On observations (simulates sensor noise) |
| **Expected single-attempt pick** | ~80% (with VLM error) | ~64% (conservative humanoid estimate) |
| **With 3 retries** | ~95% (if verification works) | ~95.3% pick; improves with more demos + Phase 3 servoing |

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Grasp verification classifier <90% accuracy | Retries become blind → reliability drops to ~90% | Collect more failure examples, augment aggressively, add motor current as second signal |
| RealSense depth noisy at range >1.5m | PCD quality degrades for distant objects | Keep manipulation within 1m reach; use colorless PCD for robustness |
| 80-100 demos insufficient per skill | Single-attempt <75% | Collect more demos; APPF pre-training helps data efficiency; humanoid needs more than fixed-base arm |
| WBC balance loss during manipulation | Robot falls during arm extension | Retrain WBC with arm perturbation domain randomization, reduce arm reach speed |
| Motor current proxy unreliable | Lose secondary grasp signal + contact controller degrades | Acceptable for grasp verification (vision is primary); calibrate per-finger on real hardware early |
| VLM task decomposition errors | Wrong skill called | Simple prompt engineering usually sufficient; add few-shot examples if needed |
| Inspire finger backlash/slop | Grasp primitive open-loop accuracy drops | Calibrate per-finger dead bands, add compliance in pre-shape |
| Contact controller bandwidth limited (50-100Hz) | Wipe/drawer force control crude | Keep lateral motion slow (5-10cm/s); fall back to position-controlled wipe at fixed Z |
| Workspace PCD crop includes irrelevant objects | DP3 confused by clutter | Tighten crop volume; use SAM2 foreground mask to filter PCD |
| APPF Phase 1 multi-skill training doesn't converge | Motion manifold too diverse across skills | Fall back to per-skill Phase 1 (smaller dataset but cleaner signal) |
| DP3 grasp trigger flag unreliable | Grasp triggered too early/late | 3-frame hysteresis; fallback to distance-based trigger |
| Head joint recording missing in teleop pipeline | PCD crop misaligned, Phase 2 training fails | Verify head joint recording BEFORE collecting new data (week 5) |
| Humanoid arm accuracy lower than fixed-base | Single-attempt pick < 80% | Plan for 80-100 demos (not 40-60); Phase 3 wrist servoing if alignment is dominant failure |