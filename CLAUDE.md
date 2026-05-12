# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ROS 2 (Jazzy) ament_cmake package `dis_tutorial3` for the FRI **RInS / Development of Intelligent Systems** course. Targets a TurtleBot 4 in Gazebo Ignition (`ros_gz`). Two course tasks live here side-by-side:

- **Task 1** (`scripts/task1.py`) — boustrophedon coverage of a small arena, YOLO+Haar face detection, ring detection by colour, Nav2-driven approach + greet via `espeak-ng`.
- **Task 2** (`scripts/task2.py`, `scripts/line_detector.py`, `scripts/face_recognizer.py`) — Industry 5.0 factory plant: don't cross yellow floor lines, follow the blue line in the second room, recognise named workers from `personnel/` portraits, dispatch barrel/anomaly inspection sub-tasks (sub-tasks merged in later).

The full Task 2 spec is `rinsTask2.pdf` (in repo root). Hardware bring-up for the physical robot is in [README.md](README.md).

## Build & run

```bash
# Build (run from your colcon workspace, with this repo at src/dis_tutorial3)
colcon build --packages-select dis_tutorial3 --symlink-install
source install/setup.bash
```

Launch chains (pick one — they're mutually exclusive on the Gazebo side):

| Goal | Launch |
|---|---|
| Sim only (no robot stack) | `ros2 launch dis_tutorial3 sim.launch.py world:=task2_yellow_demo` |
| Sim + SLAM (build a new map live) | `ros2 launch dis_tutorial3 sim_turtlebot_slam.launch.py world:=task2_yellow_demo` |
| Sim + Nav (requires a saved map) | `ros2 launch dis_tutorial3 sim_turtlebot_nav.launch.py world:=task2_yellow_demo map:=…/maps/<name>.yaml` |

Real-robot bring-up has been removed — this package is sim-only.

Run the perception/orchestration nodes in separate terminals after the launch chain is up:

```bash
ros2 run dis_tutorial3 detect_people.py
ros2 run dis_tutorial3 detect_rings.py
ros2 run dis_tutorial3 line_detector.py    # Task 2 only
ros2 run dis_tutorial3 task1.py            # OR
ros2 run dis_tutorial3 task2.py
```

The package has no test target; verification is by running the relevant launch + node combo and inspecting RViz / topics. Face recognizer can be smoke-tested standalone: `python3 scripts/face_recognizer.py personnel`.

## Architecture (the bit you need to read multiple files to see)

**Detection nodes are independent and communicate only via `MarkerArray`.** No shared blackboard, no service calls between them. Each detection node owns one ROS topic:

| Node | Publishes | Marker namespaces |
|---|---|---|
| `detect_people.py` | `/people_markers` | `faces`, `face_labels`, `face_identities` (JSON) |
| `detect_rings.py`  | `/ring_markers`   | `confirmed_rings` (color encodes ring colour) |
| `line_detector.py` | `/lines/yellow_alert` (Bool), `/lines/yellow_obstacles` (PointCloud2), `/lines/blue_target` (PoseStamped), `/lines/cell_detected` (String) |

Identity also flows through `/recognized_people` (`std_msgs/String`, JSON payload) for machine-readable consumption — same data as the `face_identities` marker namespace. Persisted to `~/.ros/known_people.json` so identities survive node restarts.

**Orchestrators (`task1.py`, `task2.py`)** subscribe to those marker topics and drive Nav2 via `goToPose`. Both subclass `RobotCommander` (`scripts/robot_commander.py`), which is the single source of truth for the Nav2 action‑client/lifecycle wait/quaternion helpers — never duplicate that wrapper.

The boustrophedon coverage path (`task1.py:_boustrophedon`) consumes the live `/map` (`OccupancyGrid` with `TRANSIENT_LOCAL` QoS), filters for clearance, and re-orders waypoints by nearest-neighbour from the spawn pose. `task2.py` re-imports `COVERAGE_SPACING` / `ROBOT_CLEARANCE` / `SWEEP_AXIS` from `task1.py` so changes apply to both.

**Yellow-line enforcement (Task 2)** is two-layered and both layers are required because we have no static map for the Task 2 world yet:

1. `line_detector.py` projects yellow pixels from the downward camera onto the floor plane and publishes them as a `PointCloud2` on `/lines/yellow_obstacles`. `config/nav2.yaml` adds this as an `observation_source` named `line_obstacles` on both the local (voxel layer) and global (obstacle layer) costmaps so the planner routes around them.
2. `task2.py` subscribes to `/lines/yellow_alert`; on `True` it cancels the Nav2 goal and reverses on `/cmd_vel_nav` until the alert clears — catches lines the planner committed to before they were visible.

## Camera-topic conventions

Two sim camera streams, both bridged in `launch/ros_gz_bridge.launch.py` — wired into the spawn chain via `launch/turtlebot4_spawn.launch.py` (uses the LOCAL bridge, not the upstream `turtlebot4_gz_bringup` one, because only the local bridge knows about the arm-mounted top_camera):

- `/oakd/rgb/preview/{image_raw,depth,depth/points,camera_info}` — forward, body-mounted Oak-D Pro. Used for face/ring detection.
- `/top_camera/rgb/preview/{image_raw,depth,depth/points,camera_info}` — arm-wrist mounted, rotated `rpy="0 -pi/2 0"` so it points downward when the arm is parked. Used for line detection (`line_detector.py` defaults to it) and for conveyor-tile inspection. Driven by the arm controller — see `scripts/arm_mover_actions.py` for predefined poses (`look_at_belt_left`, `look_at_belt_right`, `look_for_qr`, `garage`, `up`).

Physical robot uses different topics: `/gemini/color/image_raw`, `/gemini/depth/image_raw`, etc. See [README.md](README.md) for the full list. Detection scripts hard-code the sim topics; for real hardware remap with `--ros-args -r /oakd/rgb/preview/image_raw:=/gemini/color/image_raw …`.

## Personnel data

`personnel/` holds 14 portraits named `<name>_<pronouns>_<role>.png`. Pronouns ∈ `{he_him, she_her}` → gender. Filenames are the source of truth for identity metadata; `face_recognizer.py` parses them at startup and trains LBPH (primary), PCA (secondary), HOG+kNN (tiebreak) on tiny in‑process augmentations. **LBPH requires `opencv-contrib-python`**; without it the recogniser silently degrades to PCA + HOG voting.

## Adding a new perception node

The established pattern (see `detect_people.py`, `detect_rings.py`):

1. Subscribe to `/oakd/rgb/preview/image_raw` (+ depth or pointcloud) with `qos_profile_sensor_data`.
2. Use `message_filters.ApproximateTimeSynchronizer` (slop ≈ 0.05–0.1 s) when fusing RGB + depth/pointcloud.
3. Confirm a candidate over multiple frames before publishing (avoids transient false positives).
4. Project to `map` frame via `tf2_ros.Buffer.transform(PointStamped, 'map')`.
5. Publish a `MarkerArray` with stable integer IDs on a dedicated topic; orchestrators subscribe and de-duplicate by ID.

When a Task 2 sub-task (barrel inspection, anomaly model, dialogue, PDF report) is implemented, follow the same pattern: a self-contained node that publishes its results as markers or a dedicated topic, and `task2.py` consumes them in its FSM.
