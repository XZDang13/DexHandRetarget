#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


UPSTREAM_REPO = "https://github.com/fiveages-sim/robot-descriptions-common"
UPSTREAM_TAG = "v1.3.0"
RAW_BASE = "https://raw.githubusercontent.com/fiveages-sim/robot-descriptions-common"
RH56E2_XACRO_PATH = "gripper/inspire_description/xacro/hands/RH56E2.xacro"
RH56E2_CONFIG_PATH = "gripper/inspire_description/config/ros2_control/RH56E2.yaml"
LICENSE_PATH = "LICENSE"
SOURCE_PACKAGE = "gripper/inspire_description"
MESH_ROOT = "gripper/inspire_description/meshes/RH56E2"
LOCAL_XACRO_PATH = Path("xacro/hands/RH56E2.xacro")
LOCAL_CONFIG_PATH = Path("config/ros2_control/RH56E2.yaml")
LOCAL_MESH_ROOT = Path("meshes/RH56E2")
XACRO_NS = "{http://www.ros.org/wiki/xacro}"

ACTUATOR_JOINTS = (
    "thumb_joint1",
    "thumb_joint2",
    "index_joint",
    "middle_joint",
    "ring_joint",
    "pinky_joint",
)

TIP_SENSOR_JOINTS = {
    "thumb_tip": "thumb_force_sensor_4_joint",
    "index_tip": "index_force_sensor_3_joint",
    "middle_tip": "middle_force_sensor_3_joint",
    "ring_tip": "ring_force_sensor_3_joint",
    "pinky_tip": "pinky_force_sensor_3_joint",
}

EXPECTED_MIMICS = {
    "thumb_joint3": ("thumb_joint2", 0.8024),
    "thumb_joint4": ("thumb_joint2", 0.76123688),
    "index_dip": ("index_joint", 1.0843),
    "middle_dip": ("middle_joint", 1.0843),
    "ring_dip": ("ring_joint", 1.0843),
    "pinky_dip": ("pinky_joint", 1.0843),
}


@dataclass(frozen=True)
class LinkVisual:
    mesh_path: str
    scale: tuple[float, float, float]


@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float] | None
    limit: tuple[float, float] | None
    mimic: tuple[str, float] | None


@dataclass(frozen=True)
class SourceBundle:
    xacro: str
    urdf: str | None
    config: str
    license_text: str
    mesh_root: Path | None
    local_source_dir: Path | None
    urdf_file: Path | None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import real Inspire RH56E2 meshes and generate MuJoCo MJCF assets.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Local gripper/inspire_description directory to import instead of fetching the pinned upstream tag.",
    )
    parser.add_argument(
        "--urdf-file",
        type=Path,
        help="Expanded RH56E2 URDF file to import directly instead of parsing xacro.",
    )
    parser.add_argument(
        "--hand",
        choices=("left", "right"),
        help="Hand side for --urdf-file output. If omitted, infer from the URDF mesh scale.",
    )
    parser.add_argument("--assimp", default=shutil.which("assimp") or "assimp")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()
    asset_root = project_root / "assets" / "mjcf"
    mesh_dir = asset_root / "inspire_rh56e2_meshes"
    left_dir = asset_root / "inspire_rh56e2_left"
    right_dir = asset_root / "inspire_rh56e2_right"

    try:
        source = load_source(args.source_dir, args.urdf_file)
    except OSError as exc:
        print(f"Failed to load RH56E2 metadata: {exc}", file=sys.stderr)
        return 1

    if "Apache License" not in source.license_text:
        print("RH56E2 source license was not recognized as Apache-2.0", file=sys.stderr)
        return 1
    for joint in ACTUATOR_JOINTS:
        if joint not in source.config:
            print(f"RH56E2 ros2_control config does not list {joint}", file=sys.stderr)
            return 1

    written_models: list[Path] = []
    try:
        if source.urdf is not None:
            side = args.hand or infer_urdf_hand(source.urdf)
            links, joints = parse_rh56e2_urdf(source.urdf)
            validate_model_metadata(joints)
            if not args.skip_download:
                sync_and_convert_meshes(links, mesh_dir, args.assimp, source.mesh_root)
            model_dir = left_dir if side == "left" else right_dir
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "model.xml").write_text(
                generate_mjcf(side, links, joints),
                encoding="utf-8",
            )
            written_models.append(model_dir / "model.xml")
        else:
            for side, direction in (("left", 1), ("right", -1)):
                links, joints = parse_rh56e2_xacro(source.xacro, direction)
                validate_model_metadata(joints)
                if not args.skip_download:
                    sync_and_convert_meshes(links, mesh_dir, args.assimp, source.mesh_root)
                model_dir = left_dir if side == "left" else right_dir
                model_dir.mkdir(parents=True, exist_ok=True)
                (model_dir / "model.xml").write_text(
                    generate_mjcf(side, links, joints),
                    encoding="utf-8",
                )
                written_models.append(model_dir / "model.xml")
        (asset_root / "inspire_rh56e2_SOURCE.md").write_text(source_note(source), encoding="utf-8")
    except Exception as exc:
        print(f"Failed to generate RH56E2 assets: {exc}", file=sys.stderr)
        return 1

    for model_path in written_models:
        print(f"Wrote {model_path}")
    print(f"Wrote {mesh_dir}")
    return 0


def load_source(source_dir: Path | None, urdf_file: Path | None) -> SourceBundle:
    urdf_path = None if urdf_file is None else urdf_file.resolve()
    if source_dir is None:
        return SourceBundle(
            xacro="" if urdf_path is not None else fetch_text(RH56E2_XACRO_PATH),
            urdf=None if urdf_path is None else urdf_path.read_text(encoding="utf-8"),
            config=fetch_text(RH56E2_CONFIG_PATH),
            license_text=fetch_text(LICENSE_PATH),
            mesh_root=None,
            local_source_dir=None,
            urdf_file=urdf_path,
        )

    resolved = source_dir.resolve()
    xacro_path = resolved / LOCAL_XACRO_PATH
    config_path = resolved / LOCAL_CONFIG_PATH
    mesh_root = resolved / LOCAL_MESH_ROOT
    required_paths = [config_path, mesh_root]
    if urdf_path is None:
        required_paths.append(xacro_path)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    return SourceBundle(
        xacro="" if urdf_path is not None else xacro_path.read_text(encoding="utf-8"),
        urdf=None if urdf_path is None else urdf_path.read_text(encoding="utf-8"),
        config=config_path.read_text(encoding="utf-8"),
        license_text=read_local_license(resolved),
        mesh_root=mesh_root,
        local_source_dir=resolved,
        urdf_file=urdf_path,
    )


def read_local_license(source_dir: Path) -> str:
    search_roots = [source_dir, *list(source_dir.parents)[:4]]
    for root in search_roots:
        license_path = root / "LICENSE"
        if license_path.exists():
            return license_path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"LICENSE near {source_dir}")


def fetch_text(path: str) -> str:
    url = f"{RAW_BASE}/{UPSTREAM_TAG}/{path}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def fetch_bytes(path: str) -> bytes:
    url = f"{RAW_BASE}/{UPSTREAM_TAG}/{path}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def parse_rh56e2_xacro(xacro: str, direction: int) -> tuple[dict[str, LinkVisual], list[JointSpec]]:
    root = ET.fromstring(xacro)
    macro = root.find(f"{XACRO_NS}macro")
    if macro is None:
        raise ValueError("RH56E2 macro not found")
    context = {"prefix": "", "dir": -1 * int(direction), "int": int}

    links: dict[str, LinkVisual] = {}
    joints: list[JointSpec] = []
    for child in list(macro):
        tag = strip_ns(child.tag)
        if tag == "link":
            link_name = expand(child.attrib["name"], context)
            mesh = child.find("./visual/geometry/mesh")
            if mesh is None:
                continue
            filename = expand(mesh.attrib["filename"], context)
            mesh_path = filename.split("/meshes/", 1)[1]
            scale = parse_vector(mesh.attrib.get("scale", "1 1 1"), context)
            links[link_name] = LinkVisual(mesh_path=mesh_path, scale=scale)
        elif tag == "joint":
            name = expand(child.attrib["name"], context)
            joint_type = child.attrib["type"]
            parent = expand(required_child(child, "parent").attrib["link"], context)
            child_link = expand(required_child(child, "child").attrib["link"], context)
            origin = child.find("origin")
            xyz = parse_vector(origin.attrib.get("xyz", "0 0 0"), context) if origin is not None else (0.0, 0.0, 0.0)
            rpy = parse_vector(origin.attrib.get("rpy", "0 0 0"), context) if origin is not None else (0.0, 0.0, 0.0)
            axis_node = child.find("axis")
            axis = parse_vector(axis_node.attrib["xyz"], context) if axis_node is not None else None
            limit_node = child.find("limit")
            limit = None
            if limit_node is not None:
                limit = (
                    float(expand(limit_node.attrib["lower"], context)),
                    float(expand(limit_node.attrib["upper"], context)),
                )
            mimic_node = child.find("mimic")
            mimic = None
            if mimic_node is not None:
                mimic = (
                    expand(mimic_node.attrib["joint"], context),
                    float(expand(mimic_node.attrib.get("multiplier", "1"), context)),
                )
            joints.append(
                JointSpec(
                    name=name,
                    joint_type=joint_type,
                    parent=parent,
                    child=child_link,
                    xyz=xyz,
                    rpy=rpy,
                    axis=axis,
                    limit=limit,
                    mimic=mimic,
                )
            )
    return links, joints


def parse_rh56e2_urdf(urdf: str) -> tuple[dict[str, LinkVisual], list[JointSpec]]:
    root = ET.fromstring(urdf)
    links: dict[str, LinkVisual] = {}
    joints: list[JointSpec] = []

    for child in list(root):
        tag = strip_ns(child.tag)
        if tag == "link":
            link_name = child.attrib["name"]
            if link_name == "flange":
                continue
            mesh = child.find("./visual/geometry/mesh")
            if mesh is None:
                continue
            visual_origin = child.find("./visual/origin")
            if visual_origin is not None:
                xyz = parse_plain_vector(visual_origin.attrib.get("xyz", "0 0 0"))
                rpy = parse_plain_vector(visual_origin.attrib.get("rpy", "0 0 0"))
                if any(abs(value) > 1e-12 for value in (*xyz, *rpy)):
                    raise ValueError(f"Non-identity visual origin is not supported for {link_name}")
            filename = mesh.attrib["filename"]
            links[link_name] = LinkVisual(
                mesh_path=extract_mesh_path(filename),
                scale=parse_plain_vector(mesh.attrib.get("scale", "1 1 1")),
            )
        elif tag == "joint":
            if child.attrib["name"] == "flange_joint":
                continue
            origin = child.find("origin")
            axis_node = child.find("axis")
            limit_node = child.find("limit")
            mimic_node = child.find("mimic")
            joints.append(
                JointSpec(
                    name=child.attrib["name"],
                    joint_type=child.attrib["type"],
                    parent=required_child(child, "parent").attrib["link"],
                    child=required_child(child, "child").attrib["link"],
                    xyz=parse_plain_vector(origin.attrib.get("xyz", "0 0 0")) if origin is not None else (0.0, 0.0, 0.0),
                    rpy=parse_plain_vector(origin.attrib.get("rpy", "0 0 0")) if origin is not None else (0.0, 0.0, 0.0),
                    axis=parse_plain_vector(axis_node.attrib["xyz"]) if axis_node is not None else None,
                    limit=(
                        float(limit_node.attrib["lower"]),
                        float(limit_node.attrib["upper"]),
                    )
                    if limit_node is not None
                    else None,
                    mimic=(
                        mimic_node.attrib["joint"],
                        float(mimic_node.attrib.get("multiplier", "1")),
                    )
                    if mimic_node is not None
                    else None,
                )
            )
    return links, joints


def infer_urdf_hand(urdf: str) -> str:
    root = ET.fromstring(urdf)
    mesh = root.find(".//mesh")
    if mesh is None:
        raise ValueError("--hand is required because the URDF has no mesh scale to infer side")
    scale = parse_plain_vector(mesh.attrib.get("scale", "1 1 1"))
    return "left" if scale[1] < 0.0 else "right"


def validate_model_metadata(joints: list[JointSpec]) -> None:
    by_name = {joint.name: joint for joint in joints}
    for name in ACTUATOR_JOINTS:
        joint = by_name.get(name)
        if joint is None:
            raise ValueError(f"Missing actuator joint {name}")
        if joint.limit is None:
            raise ValueError(f"Missing limit for actuator joint {name}")
    for mimic_name, expected in EXPECTED_MIMICS.items():
        mimic_joint = by_name.get(mimic_name)
        if mimic_joint is None or mimic_joint.mimic is None:
            raise ValueError(f"Missing mimic joint metadata for {mimic_name}")
        target, multiplier = expected
        actual_target, actual_multiplier = mimic_joint.mimic
        if actual_target != target or abs(actual_multiplier - multiplier) > 1e-8:
            raise ValueError(f"Unexpected mimic for {mimic_name}: {mimic_joint.mimic}")


def sync_and_convert_meshes(
    links: dict[str, LinkVisual],
    mesh_dir: Path,
    assimp: str,
    local_mesh_root: Path | None,
) -> None:
    mesh_dir.mkdir(parents=True, exist_ok=True)
    needed = sorted({visual.mesh_path for visual in links.values()})
    for mesh_path in needed:
        source_rel = Path(mesh_path).relative_to("RH56E2")
        source_path = f"{MESH_ROOT}/{source_rel}"
        suffix = Path(mesh_path).suffix.lower()
        if suffix == ".glb":
            local_glb = mesh_dir / mesh_path
            local_obj = local_glb.with_suffix(".obj")
            local_obj.parent.mkdir(parents=True, exist_ok=True)
            if local_mesh_root is None:
                if not local_glb.exists():
                    local_glb.write_bytes(fetch_bytes(source_path))
            else:
                shutil.copy2(local_mesh_root / source_rel, local_glb)
            if not local_glb.exists():
                local_glb.write_bytes(fetch_bytes(source_path))
            subprocess.run(
                [assimp, "export", str(local_glb), str(local_obj), "-f", "obj"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        elif suffix == ".stl":
            local_stl = mesh_dir / mesh_path
            local_stl.parent.mkdir(parents=True, exist_ok=True)
            if local_mesh_root is None:
                if not local_stl.exists():
                    local_stl.write_bytes(fetch_bytes(source_path))
            else:
                shutil.copy2(local_mesh_root / source_rel, local_stl)
            if not local_stl.exists():
                local_stl.write_bytes(fetch_bytes(source_path))
        else:
            raise ValueError(f"Unsupported RH56E2 mesh format: {mesh_path}")


def generate_mjcf(side: str, links: dict[str, LinkVisual], joints: list[JointSpec]) -> str:
    children_by_parent: dict[str, list[JointSpec]] = {}
    joints_by_name = {joint.name: joint for joint in joints}
    for joint in joints:
        if joint.name == "flange_joint":
            continue
        children_by_parent.setdefault(joint.parent, []).append(joint)

    tip_sites: dict[str, tuple[str, tuple[float, float, float]]] = {}
    for site_name, joint_name in TIP_SENSOR_JOINTS.items():
        joint = joints_by_name[joint_name]
        tip_sites[site_name] = (joint.parent, joint.xyz)

    mesh_assets: list[str] = []
    seen_assets: set[str] = set()
    for visual in sorted({visual for visual in links.values()}, key=lambda item: item.mesh_path):
        mesh_name = mesh_asset_name(visual.mesh_path)
        if mesh_name in seen_assets:
            continue
        seen_assets.add(mesh_name)
        file_path = mesh_file_for_mujoco(visual.mesh_path)
        scale = fmt_vector(visual.scale)
        mesh_assets.append(f'    <mesh name="{mesh_name}" file="{file_path}" scale="{scale}"/>')

    lines: list[str] = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<mujoco model="inspire_rh56e2_{side}">',
        '  <compiler angle="radian" meshdir="../inspire_rh56e2_meshes" autolimits="true" balanceinertia="true"/>',
        '  <default>',
        '    <joint damping="0.03" armature="0.0001"/>',
        '    <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.78 0.80 0.82 1"/>',
        '    <site type="sphere" size="0.001" rgba="1 0.12 0.08 0"/>',
        '  </default>',
        '  <asset>',
        *mesh_assets,
        '  </asset>',
        '  <worldbody>',
        '    <body name="hand_root">',
    ]
    add_link_geoms(lines, "hand_base", links, indent=6)
    add_tip_sites(lines, "hand_base", tip_sites, indent=6)
    for joint in children_by_parent.get("hand_base", []):
        append_joint_body(lines, joint, children_by_parent, links, tip_sites, indent=6)
    lines.extend(
        [
            '    </body>',
            '  </worldbody>',
            '  <actuator>',
        ]
    )
    for name in ACTUATOR_JOINTS:
        joint = joints_by_name[name]
        assert joint.limit is not None
        lines.append(
            f'    <position name="act_{name}" joint="{name}" '
            f'ctrlrange="{fmt_float(joint.limit[0])} {fmt_float(joint.limit[1])}" kp="10"/>'
        )
    lines.extend(['  </actuator>', '  <equality>'])
    for name, (target, multiplier) in EXPECTED_MIMICS.items():
        lines.append(
            f'    <joint joint1="{name}" joint2="{target}" polycoef="0 {fmt_float(multiplier)} 0 0 0"/>'
        )
    lines.extend(['  </equality>', '</mujoco>', ''])
    return "\n".join(lines)


def append_joint_body(
    lines: list[str],
    joint: JointSpec,
    children_by_parent: dict[str, list[JointSpec]],
    links: dict[str, LinkVisual],
    tip_sites: dict[str, tuple[str, tuple[float, float, float]]],
    *,
    indent: int,
) -> None:
    spaces = " " * indent
    attrs = [
        f'name="{joint.child}"',
        f'pos="{fmt_vector(joint.xyz)}"',
    ]
    if any(abs(value) > 1e-12 for value in joint.rpy):
        attrs.append(f'quat="{fmt_quat(quat_from_urdf_rpy(joint.rpy))}"')
    lines.append(f"{spaces}<body {' '.join(attrs)}>")
    lines.append(f'{spaces}  <inertial pos="0 0 0" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>')
    if joint.joint_type != "fixed":
        if joint.axis is None or joint.limit is None:
            raise ValueError(f"Movable joint {joint.name} is missing axis or limit")
        lines.append(
            f'{spaces}  <joint name="{joint.name}" axis="{fmt_vector(joint.axis)}" '
            f'range="{fmt_float(joint.limit[0])} {fmt_float(joint.limit[1])}"/>'
        )
    add_link_geoms(lines, joint.child, links, indent=indent + 2)
    add_tip_sites(lines, joint.child, tip_sites, indent=indent + 2)
    for child_joint in children_by_parent.get(joint.child, []):
        append_joint_body(lines, child_joint, children_by_parent, links, tip_sites, indent=indent + 2)
    lines.append(f"{spaces}</body>")


def add_link_geoms(lines: list[str], link_name: str, links: dict[str, LinkVisual], *, indent: int) -> None:
    visual = links.get(link_name)
    if visual is None:
        return
    spaces = " " * indent
    mesh_name = mesh_asset_name(visual.mesh_path)
    color = '0.46 0.48 0.50 1' if "sensor" in visual.mesh_path else '0.78 0.80 0.82 1'
    lines.append(f'{spaces}<geom name="{safe_name(link_name)}_geom" mesh="{mesh_name}" rgba="{color}"/>')


def add_tip_sites(
    lines: list[str],
    link_name: str,
    tip_sites: dict[str, tuple[str, tuple[float, float, float]]],
    *,
    indent: int,
) -> None:
    spaces = " " * indent
    for site_name, (parent, xyz) in tip_sites.items():
        if parent == link_name:
            lines.append(f'{spaces}<site name="{site_name}" pos="{fmt_vector(xyz)}"/>')


def required_child(node: ET.Element, name: str) -> ET.Element:
    child = node.find(name)
    if child is None:
        raise ValueError(f"Missing child {name} in {node.tag}")
    return child


def strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def expand(value: str, context: dict[str, object]) -> str:
    def replace(match: re.Match[str]) -> str:
        expr = match.group(1)
        return str(eval(expr, {"__builtins__": {}}, context))

    return re.sub(r"\$\{([^}]+)\}", replace, value)


def parse_vector(value: str, context: dict[str, object]) -> tuple[float, float, float]:
    expanded = expand(value, context)
    parts = [float(part) for part in expanded.split()]
    if len(parts) != 3:
        raise ValueError(f"Expected 3-vector, got {value!r} -> {expanded!r}")
    return parts[0], parts[1], parts[2]


def parse_plain_vector(value: str) -> tuple[float, float, float]:
    parts = [float(part) for part in value.split()]
    if len(parts) != 3:
        raise ValueError(f"Expected 3-vector, got {value!r}")
    return parts[0], parts[1], parts[2]


def extract_mesh_path(filename: str) -> str:
    if "/meshes/" not in filename:
        raise ValueError(f"RH56E2 mesh path does not contain /meshes/: {filename}")
    return filename.split("/meshes/", 1)[1]


def mesh_asset_name(mesh_path: str) -> str:
    path = Path(mesh_path)
    rel = path.with_suffix(".obj") if path.suffix.lower() == ".glb" else path
    return safe_name(str(rel.with_suffix("")))


def mesh_file_for_mujoco(mesh_path: str) -> str:
    path = Path(mesh_path)
    if path.suffix.lower() == ".glb":
        return str(path.with_suffix(".obj"))
    return str(path)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def fmt_vector(vector: tuple[float, float, float]) -> str:
    return " ".join(fmt_float(value) for value in vector)


def fmt_quat(quat: tuple[float, float, float, float]) -> str:
    return " ".join(fmt_float(value) for value in quat)


def fmt_float(value: float) -> str:
    text = f"{float(value):.10g}"
    return "0" if text == "-0" else text


def quat_from_urdf_rpy(rpy: tuple[float, float, float]) -> tuple[float, float, float, float]:
    roll, pitch, yaw = rpy
    return normalize_quat(
        quat_multiply(
            quat_multiply(axis_angle_quat("z", yaw), axis_angle_quat("y", pitch)),
            axis_angle_quat("x", roll),
        )
    )


def axis_angle_quat(axis: str, angle: float) -> tuple[float, float, float, float]:
    half = angle * 0.5
    sine = math.sin(half)
    if axis == "x":
        return math.cos(half), sine, 0.0, 0.0
    if axis == "y":
        return math.cos(half), 0.0, sine, 0.0
    if axis == "z":
        return math.cos(half), 0.0, 0.0, sine
    raise ValueError(f"Unsupported quaternion axis: {axis}")


def quat_multiply(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = second
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def normalize_quat(quat: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in quat))
    if norm <= 0.0:
        raise ValueError("Quaternion norm is zero")
    normalized = tuple(value / norm for value in quat)
    assert len(normalized) == 4
    return normalized


def source_note(source: SourceBundle) -> str:
    local_line = ""
    if source.local_source_dir is not None:
        local_line = f"- Local import source: `{source.local_source_dir}`\n"
    urdf_line = ""
    if source.urdf_file is not None:
        urdf_line = f"- Expanded URDF source: `{source.urdf_file}`\n"
    source_detail_lines = (local_line + urdf_line).rstrip()
    source_files = (
        "- Source files: expanded RH56E2 URDF, `config/ros2_control/RH56E2.yaml`"
        if source.urdf_file is not None
        else "- Source files: `xacro/hands/RH56E2.xacro`, `config/ros2_control/RH56E2.yaml`"
    )
    return f"""# Inspire RH56E2 Asset Source

These MuJoCo assets are generated from the real RH56E2 description and mesh
files:

- Repository: {UPSTREAM_REPO}
- Tag: `{UPSTREAM_TAG}`
- Source package: `{SOURCE_PACKAGE}`
- Import mode: `{"expanded URDF" if source.urdf_file is not None else ("local source directory" if source.local_source_dir is not None else "pinned upstream fetch")}`
{source_detail_lines}
{source_files}
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
"""


if __name__ == "__main__":
    raise SystemExit(main())
