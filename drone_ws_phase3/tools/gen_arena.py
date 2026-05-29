#!/usr/bin/python3
"""
Phase 3 — Arena SDF generator.

Reads an arena JSON config (see arenas/SCHEMA.md) and emits a Gazebo SDF
world file:
  - default sun + atmosphere
  - large textured ground_plane (mars_sand PBR — proven to render in ogre2)
  - polygon tape boundary (one thin yellow box per edge)
  - walls (one box per edge, taller, slightly outside the tape)
  - optional floor-marker grid clipped to the polygon
  - red landing pad at config.base
  - x500_flow include with spawn pose

Usage:
  python3 tools/gen_arena.py arenas/<name>.json  →  worlds/<name>.sdf
"""
from __future__ import annotations
import json
import math
import os
import sys
from pathlib import Path

# ── World pieces (constants) ───────────────────────────────────────────────
GROUND_TEXTURE = ('file:///home/vishnu/drone_ws/PX4-Autopilot/'
                  'Tools/simulation/gz/worlds/mars_sand.png')
TAPE_W   = 0.06   # m  tape width (perpendicular to edge)
TAPE_T   = 0.01   # m  tape thickness (z)
TAPE_Z   = 0.005  # m  tape top-of-floor offset
WALL_T   = 0.10   # m  wall thickness
WALL_OFFSET = 0.50  # m  walls sit this far OUTSIDE the tape
MARKER_SIZE = 0.20  # m  square marker side
MARKER_Z    = 0.002 # m
PAD_SIZE = 0.50   # m  red landing pad side
PAD_Z    = 0.005


# ── Polygon helpers ────────────────────────────────────────────────────────
def polygon_signed_area(poly):
    n = len(poly); a = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return a * 0.5


def ensure_ccw(poly):
    return poly if polygon_signed_area(poly) > 0 else list(reversed(poly))


def point_in_poly(x, y, poly):
    """Standard ray-cast."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


# ── SDF fragment builders ──────────────────────────────────────────────────
def sun_atmosphere_physics():
    # NOTE: plugins are loaded globally by gz-sim's server config — DO NOT
    # duplicate them here or sensor publishing breaks (Preflight Fail: ekf2
    # missing data). The phase-2 working SDF had zero <plugin> tags.
    return r'''    <physics type="ode">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>250</real_time_update_rate>
    </physics>
    <gravity>0 0 -9.8</gravity>
    <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
    <atmosphere type="adiabatic"/>
    <scene>
      <grid>false</grid>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.7 0.7 0.7 1</background>
      <shadows>true</shadows>
    </scene>
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>47.397971057728974</latitude_deg>
      <longitude_deg>8.546163739800146</longitude_deg>
      <elevation>488.0</elevation>
      <heading_deg>0</heading_deg>
    </spherical_coordinates>
    <light name="sunUTC" type="directional">
      <pose>0 0 500 0 0 0</pose>
      <cast_shadows>true</cast_shadows>
      <intensity>1</intensity>
      <direction>0.001 0.625 -0.78</direction>
      <diffuse>0.904 0.904 0.904 1</diffuse>
      <specular>0.271 0.271 0.271 1</specular>
    </light>'''


def ground_plane():
    return f'''    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>1 1</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>500 500</size></plane></geometry>
          <material>
            <ambient>0.70 0.40 0.22 1</ambient>
            <diffuse>0.75 0.42 0.22 1</diffuse>
            <specular>0.04 0.02 0.01 1</specular>
            <pbr>
              <metal>
                <albedo_map>{GROUND_TEXTURE}</albedo_map>
                <roughness>0.95</roughness>
                <metalness>0.0</metalness>
              </metal>
            </pbr>
          </material>
        </visual>
      </link>
    </model>'''


def edge_box(p0, p1, width, thickness, z_top, color, name, outward_offset=0.0):
    """Emit an SDF box visual along an edge from p0 to p1."""
    x0, y0 = p0; x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return ''
    ux, uy = dx / L, dy / L
    # outward normal (rotate 90° CW for a CCW polygon → pointing OUTSIDE)
    nx, ny = uy, -ux
    cx = (x0 + x1) / 2 + outward_offset * nx
    cy = (y0 + y1) / 2 + outward_offset * ny
    yaw = math.atan2(dy, dx)
    sx = L + width  # extend a hair to overlap at corners
    sy = width
    sz = thickness
    r, g, b, a = color
    return f'''    <model name="{name}">
      <static>true</static>
      <pose>{cx:.4f} {cy:.4f} {z_top:.4f} 0 0 {yaw:.4f}</pose>
      <link name="link">
        <visual name="visual">
          <geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>
          <material>
            <ambient>{r} {g} {b} {a}</ambient>
            <diffuse>{r} {g} {b} {a}</diffuse>
          </material>
        </visual>
        <collision name="collision">
          <geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>
        </collision>
      </link>
    </model>'''


def tape_models(polygon, color):
    out = []
    for i in range(len(polygon)):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % len(polygon)]
        out.append(edge_box(p0, p1, TAPE_W, TAPE_T, TAPE_Z,
                            color, f'tape_{i}', outward_offset=0.0))
    return '\n'.join(out)


def wall_models(polygon, height):
    grey = (0.5, 0.5, 0.5, 1)
    out = []
    for i in range(len(polygon)):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % len(polygon)]
        out.append(edge_box(p0, p1, WALL_T, height, height / 2,
                            grey, f'wall_{i}', outward_offset=WALL_OFFSET))
    return '\n'.join(out)


def marker_grid(polygon, res):
    if res <= 0:
        return ''
    xs = [p[0] for p in polygon]; ys = [p[1] for p in polygon]
    xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
    # Start grid aligned to half-integer-of-res, like the original markers
    x = math.floor(xmin / res) * res + res / 2.0
    out = []
    idx = 0
    while x <= xmax:
        y = math.floor(ymin / res) * res + res / 2.0
        while y <= ymax:
            if point_in_poly(x, y, polygon):
                out.append(
                    f'    <model name="ft_{idx}">'
                    f'<static>true</static>'
                    f'<pose>{x:.3f} {y:.3f} {MARKER_Z} 0 0 0</pose>'
                    f'<link name="l"><visual name="v">'
                    f'<geometry><box><size>{MARKER_SIZE} {MARKER_SIZE} 0.002</size></box></geometry>'
                    f'<material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse></material>'
                    f'</visual></link></model>')
                idx += 1
            y += res
        x += res
    return '\n'.join(out)


def landing_pad(base):
    bx, by = base
    return f'''    <model name="landing_pad">
      <static>true</static>
      <pose>{bx:.3f} {by:.3f} {PAD_Z} 0 0 0</pose>
      <link name="link">
        <visual name="pad_visual">
          <geometry><box><size>{PAD_SIZE} {PAD_SIZE} 0.005</size></box></geometry>
          <material>
            <ambient>1.0 0.0 0.0 1</ambient>
            <diffuse>1.0 0.0 0.0 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>'''


def x500_flow_include(spawn, yaw):
    sx, sy, sz = spawn
    return f'''    <include>
      <uri>model://x500_flow</uri>
      <name>x500_flow_0</name>
      <pose>{sx:.3f} {sy:.3f} {sz:.3f} 0 0 {yaw:.3f}</pose>
    </include>'''


# ── Main ───────────────────────────────────────────────────────────────────
def build_sdf(cfg, world_name):
    polygon = ensure_ccw([tuple(p) for p in cfg['polygon']])
    if not 3 <= len(polygon) <= 6:
        raise SystemExit(f"polygon must have 3..6 vertices (got {len(polygon)})")
    if not point_in_poly(cfg['base'][0], cfg['base'][1], polygon):
        raise SystemExit("base point is outside the polygon")

    parts = [
        f'<?xml version="1.0" ?>',
        f'<sdf version="1.10">',
        f'  <world name="{world_name}">',
        sun_atmosphere_physics(),
        ground_plane(),
        tape_models(polygon, cfg.get('tape_color', [1, 0.9, 0, 1])),
        wall_models(polygon, cfg.get('wall_height', 4.5)),
        marker_grid(polygon, cfg.get('marker_grid_res', 1.0)),
        landing_pad(cfg['base']),
        # NOTE: drone is spawned by PX4 via PX4_GZ_MODEL_POSE env var.
        # Do NOT include the model here or it conflicts with PX4's spawn.
        '  </world>',
        '</sdf>',
    ]
    return '\n'.join(parts) + '\n'


def main(argv):
    if len(argv) != 2:
        print("usage: gen_arena.py <arena_config.json>", file=sys.stderr)
        sys.exit(2)
    cfg_path = Path(argv[1]).resolve()
    cfg = json.loads(cfg_path.read_text())
    name = cfg['name']

    # Always emit to the SAME PX4 worlds dir so gz finds it
    out_dir = Path('/home/vishnu/drone_ws/PX4-Autopilot/'
                   'Tools/simulation/gz/worlds')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{name}.sdf'
    out_path.write_text(build_sdf(cfg, name))

    # Also drop a copy under phase3 worlds/ for inspection
    local_dir = Path('/home/vishnu/drone_ws_phase3/worlds')
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / f'{name}.sdf').write_text((out_dir / f'{name}.sdf').read_text())

    print(f'OK: {name} → {out_path}')


if __name__ == '__main__':
    main(sys.argv)
