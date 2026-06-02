# DexHandRetarget

DexHandRetarget is the standalone PC-side project for receiving Quest 3 hand
tracking frames and retargeting one selected hand to MuJoCo dexterous hand
models.

## Current Scope

- HTTPS WebRTC signaling endpoint for Quest WebRTC clients.
- WebRTC DataChannel receiver for `hand_skeleton_frame` JSON.
- Quest frames use the Quest/OpenXR 26-joint skeleton.
- MuJoCo + SciPy IK retargeting to Inspire DFQ and Inspire RH56E2 left/right MJCF models.
- JSONL output for 6-value dexterous hand actuator commands.
- JSONL replay recording and offline playback for retarget tuning.
- Latest-frame state cache and receive FPS statistics.
- Matplotlib 3D skeleton viewer for debugging left/right hands.

## Setup

```bash
cd DexHandRetarget
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For command-line entrypoint development, install the package in editable mode:

```bash
python -m pip install -e .
```

## Run

```bash
python server.py --host 0.0.0.0 --https
# or, after `pip install -e .`
dexhand-retarget --host 0.0.0.0 --https
```

The server prints a client URL such as:

```text
Client URL:  https://192.168.1.101:8443
```

It also prints `Server IP` / `Server IPs` explicitly at startup, which is the
address to enter from the Quest network.

In the Quest app, enter only the `IP:port` part if the Unity UI already shows
the `https://` prefix.

## Options

```bash
python server.py --https --draw-axes
python server.py --host 0.0.0.0 --port 8443 --https
```

`--draw-axes` draws wrist and palm orientation axes from the received
quaternions. The default view draws joint points and bone lines only.

## Replay

Record Quest WebRTC frames while running live:

```bash
python server.py --https --save-replay replays/quest_session.jsonl
python server.py --https --retarget-model dfq --hand right --save-replay replays/right_hand.jsonl
python server.py --https --retarget-model rh56e2 --hand right --save-replay replays/right_hand.jsonl
```

The replay file is JSONL. Each line stores the receive elapsed time and the
original `hand_skeleton_frame` payload, so later runs can reproduce the same
input without connecting the headset.

Play a replay into the normal skeleton or retarget viewer:

```bash
python server.py --replay replays/right_hand.jsonl
python server.py --retarget-model dfq --hand right --replay replays/right_hand.jsonl
python server.py --retarget-model rh56e2 --hand right --replay replays/right_hand.jsonl
```

For faster retarget iteration without opening a viewer, process every replay
frame and emit retarget command JSONL:

```bash
python server.py --retarget-model dfq --hand right --replay replays/right_hand.jsonl --replay-headless
python server.py --retarget-model rh56e2 --hand right --replay replays/right_hand.jsonl --replay-headless
```

Use `--replay-speed 2.0` to play recorded timing at 2x speed, `--replay-fps 90`
to force a fixed playback rate, and `--replay-loop` to loop while a viewer is
open.

## Replay Evaluation

Evaluate a saved replay against the selected retargeter without opening a
viewer:

```bash
python server.py --eval-replay replays/right_hand.jsonl --hand right
python server.py --eval-replay replays/right_hand.jsonl --hand right --eval-output report.json
python server.py --eval-replay replays/right_hand.jsonl --retarget-model rh56e2 --hand right
```

The evaluator prints a compact summary and, when `--eval-output` is provided,
writes a JSON report with coverage, thumb geometry errors, command smoothness,
estimated thumb yaw/pitch lag, and per-frame thumb details. Use `--eval-stride`
and `--eval-max-frames` for quick parameter sweeps:

```bash
python server.py --eval-replay replays/right_hand.jsonl --hand right --eval-stride 10
python server.py --eval-replay replays/right_hand.jsonl --hand right --retarget-alpha 0.0 --retarget-max-nfev 15
```

## Retarget Models

Run the receiver with a MuJoCo retarget viewer:

```bash
python server.py --https --retarget-model dfq --hand right
python server.py --https --retarget-model rh56e2 --hand right
```

`--retarget-dfq` remains as a backward-compatible alias for
`--retarget-model dfq`. Use `--hand left` for left-hand models. By default the
model path is selected from:

```text
assets/mjcf/inspire_dfq_right/model.xml
assets/mjcf/inspire_dfq_left/model.xml
assets/mjcf/inspire_rh56e2_right/model.xml
assets/mjcf/inspire_rh56e2_left/model.xml
```

Override it with `--model-path` when testing another MJCF. `--dfq-model-path`
is still accepted as a DFQ-only alias. The retarget mode prints one compact JSON
object per command to stdout:

```json
{"type":"retarget_command","version":1,"sequence":123,"timestamp":1.23,"hand":"Right","backend":"rh56e2","model":"inspire_rh56e2_right","mode":"tracking","ctrl":[0.0,0.0,0.4,0.3,0.2,0.1],"actuators":[{"name":"act_index_joint","joint":"index_joint","value":0.4}],"loss":0.03}
```

When `--command-output stdout` is active, receiver logs are written to stderr so
stdout can be piped into another process. Use `--command-output off` to disable
command printing.

If the selected Quest hand is lost or missing required joints, retargeting holds the last
successful tracking pose. Before the first valid tracking frame arrives, the model
stays in the clipped open pose and emits `mode="waiting"`.

Retargeting uses Quest finger bend as a strong prior and MuJoCo IK as a
correction step. Use `--retarget-alpha` to tune command smoothing; lower values
respond faster. Use `--retarget-max-nfev` to tune the per-frame IK iteration
budget. Use `--retarget-max-step` to cap per-actuator ctrl changes per frame;
the default is `0.16`, and `0` disables the rate limiter. Use
`--retarget-release-max-step` to slow non-thumb finger opening separately from
closing; the default `0.12` helps prevent a held grasp from opening on brief
Quest tracking jitter. Use `--retarget-feature-alpha` and
`--retarget-feature-deadband` to smooth Quest palm-frame finger features before
IK; defaults are `0.35` and `0.015`.

## RH56E2 Assets

Packaged RH56E2 MJCF assets can be regenerated from the local
`gripper/inspire_description` source package:

```bash
python scripts/import_rh56e2_assets.py \
  --source-dir /Users/xdang/Downloads/robot-descriptions-common-main/gripper/inspire_description
```

An already-expanded RH56E2 URDF can also be used for one side. The importer
uses the URDF joint tree and limits, skips the optional `flange` root, converts
GLB meshes to OBJ, and infers the side from the mesh scale unless `--hand` is
provided:

```bash
python scripts/import_rh56e2_assets.py \
  --urdf-file /Users/xdang/Downloads/RH56E2.urdf \
  --source-dir /Users/xdang/Downloads/robot-descriptions-common-main/gripper/inspire_description
```

## Planned Next Layer

The next version can add the hardware driver stack as separate modules:

```text
Quest WebRTC JSON
  -> HandSkeletonFrame parser
  -> hand coordinate normalization
  -> MuJoCo hand IK
  -> command smoothing and JSONL output
  -> hardware driver
```
