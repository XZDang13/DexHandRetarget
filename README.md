# DexHandRetarget

DexHandRetarget is the standalone PC-side project for receiving Quest 3 hand
tracking frames and retargeting one selected hand to an Inspire DFQ MuJoCo
model.

## Current Scope

- HTTPS WebRTC signaling endpoint for Quest WebRTC clients.
- WebRTC DataChannel receiver for `hand_skeleton_frame` JSON.
- Quest frames use the Quest/OpenXR 26-joint skeleton.
- MuJoCo + SciPy IK retargeting to Inspire DFQ left/right MJCF models.
- JSONL output for the 6-value Inspire DFQ actuator command.
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
python server.py --https --retarget-dfq --hand right --save-replay replays/right_hand.jsonl
```

The replay file is JSONL. Each line stores the receive elapsed time and the
original `hand_skeleton_frame` payload, so later runs can reproduce the same
input without connecting the headset.

Play a replay into the normal skeleton or DFQ viewer:

```bash
python server.py --replay replays/right_hand.jsonl
python server.py --retarget-dfq --hand right --replay replays/right_hand.jsonl
```

For faster retarget iteration without opening a viewer, process every replay
frame and emit DFQ command JSONL:

```bash
python server.py --retarget-dfq --hand right --replay replays/right_hand.jsonl --replay-headless
```

Use `--replay-speed 2.0` to play recorded timing at 2x speed, `--replay-fps 90`
to force a fixed playback rate, and `--replay-loop` to loop while a viewer is
open.

## DFQ Retarget

Run the receiver with the MuJoCo DFQ retarget viewer:

```bash
python server.py --https --retarget-dfq --hand right
```

Use `--hand left` for the left DFQ model. By default the model path is selected
from:

```text
assets/mjcf/inspire_dfq_right/model.xml
assets/mjcf/inspire_dfq_left/model.xml
```

Override it with `--dfq-model-path` when testing another MJCF. The retarget mode
prints one compact JSON object per command to stdout:

```json
{"type":"dfq_command","version":1,"sequence":123,"timestamp":1.23,"hand":"Right","model":"inspire_dfq_right","mode":"tracking","ctrl":[0.0,0.0,0.4,0.3,0.2,0.1],"actuators":[{"name":"act_R_index_proximal_joint","joint":"R_index_proximal_joint","value":0.4}],"loss":0.03}
```

When `--command-output stdout` is active, receiver logs are written to stderr so
stdout can be piped into another process. Use `--command-output off` to disable
command printing.

If the selected Quest hand is lost or missing required joints, DFQ holds the last
successful tracking pose. Before the first valid tracking frame arrives, DFQ
stays in the clipped open pose and emits `mode="waiting"`.

Retargeting uses Quest finger bend as a strong prior and MuJoCo IK as a
correction step. Use `--retarget-alpha` to tune command smoothing; lower values
respond faster. Use `--retarget-max-nfev` to tune the per-frame IK iteration
budget.

## Planned Next Layer

The next version can add the hardware driver stack as separate modules:

```text
Quest WebRTC JSON
  -> HandSkeletonFrame parser
  -> hand coordinate normalization
  -> Inspire DFQ MuJoCo IK
  -> command smoothing and JSONL output
  -> Inspire DFQ driver
```
