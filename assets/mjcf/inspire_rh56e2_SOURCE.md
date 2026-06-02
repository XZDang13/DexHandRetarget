# Inspire RH56E2 Asset Source

These MuJoCo assets are generated from the real RH56E2 description and mesh
files:

- Repository: https://github.com/fiveages-sim/robot-descriptions-common
- Tag: `v1.3.0`
- Source package: `gripper/inspire_description`
- Import mode: `local source directory`
- Local import source: `/Users/xdang/Downloads/robot-descriptions-common-main/gripper/inspire_description`
- Source files: `xacro/hands/RH56E2.xacro`, `config/ros2_control/RH56E2.yaml`
- Mesh source: `meshes/RH56E2/*.glb` and `meshes/RH56E2/sensors/*.STL`
- License: Apache License 2.0

The upstream GLB meshes are converted to OBJ with Assimp because MuJoCo loads
OBJ/STL meshes directly. The generated MJCF preserves the RH56E2 link tree,
visual mesh transforms, six position-controlled joints, joint limits, and mimic
ratios used by the source URDF.

Control order:

```text
thumb_joint1
thumb_joint2
index_joint
middle_joint
ring_joint
pinky_joint
```
