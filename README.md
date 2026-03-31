# The Clickbait Paradigm

Clickbait is a behavioral neuroscience paradigm in which freely-moving rodents are trained to search for hidden targets in an arena. These targets are defined by a spatial grid overlaid on realtime video of the arena (visible to the researcher, but not the mouse), and the objective is for the mouse to explore the arena until it occupies a target cell in the grid. Upon occupying the target location, the mouse hears an audible "click" from a solenoid, which indicates that a water reward is available. The mouse then drinks the reward, which initiates a new trial.

Target locations are drawn from configurable spatial probability distributions (normal or log-normal), allowing the experimenter to bias where in the arena rewards appear across a session. An optional **flip-state** mechanism alternates between two distributions after a randomized number of trials, enabling within-session reversals of the spatial reward landscape.

## Requirements

- [Bonsai-rx](https://bonsai-rx.org/) with the following packages:
  - `Bonsai.Arduino` — digital/analog I/O for reward delivery and sensors
  - `Bonsai.Scripting.IronPython` — embedded Python task logic
  - `Bonsai.Vision` / `OpenCV.Net` — image acquisition and processing
  - `BonVision` — Visual stimulus presentation
  - `Bonsai.Pylon` or `Bonsai.PointGrey` — camera drivers (rig-dependent)
  - `Bonsai.DAQmx` — analog input for sniff/poke sensors
- Arduino microcontroller (water solenoids, air valves, analog sensors)
- Compatible camera (Basler Pylon or FLIR/PointGrey)
- Optional: [SLEAP](https://sleap.ai/) for markerless pose estimation

## Workflows

### Core Task Variants

| Directory | Description |
|---|---|
| `clickbait_master/` | Master workflow. Captures, preprocesses, and tracks centroid position in Bonsai. Uses a 2d Gaussian distribution for target locations; auto-stops on session timer expiry. |
| `clickbait_motivate/` | Primary 'motivation' task. Targets drawn from a 2D log-normal distribution biased toward one end of the arena. Supports flip-state transitions, flipping the peak of log-normal distribution to the opposite end of the arena. |
| `clickbait_motivate_sleap/` | SLEAP-based variant of `clickbait_motivate`. Replaces thresholding-based centroid detection with pose-estimation model trained with SLEAP |
| `clickbait_motivate_sleap_timer/` | Adds a per-trial timeout (default 5 s) and incorrect-trial counter to the SLEAP variant. Uses a smaller 3×7 grid. |
| `clickbait_odor/` | Odor-cued spatial task. Same grid-based reward structure but targets drawn from a standard normal distribution; no flip states. |
| `clickbait_behavior_3port/` | Three-port variant with DAQmx analog input for sniff and nose-poke sensors, PointGrey camera, and a grid-based task with maze-coordinate logic. For use with FMON rig.|
| `clickbait_dry_morris_maze/` | Virtual Morris maze variant. Overlays a navigable grid on the arena and visualizes a target probability distribution without water reward delivery. |


### Utilities

| Directory | Description |
|---|---|
| `water_calibration/` | Pulses left and right water valves (100×, 32 ms open, 500 ms ITI) to measure delivered volume. |
| `water_flush/` | Opens water valves when physical button is pressed. |
| `utils/fan_test/` | Sets analog and digital outputs on an Arduino to test fan speed control. |
| `utils/pin_test/` | Tests digital output on left and right air-valve pins. |

### Work in Progress

| Directory | Description |
|---|---|
| `wip/clickbait-refactor-063025/` | Modular refactor of the monolithic Clickbait Task script into discrete reactive Bonsai sub-workflows (state coordination, search, reward, withdrawal, ITI, target generation). See `PHASE2_MODULAR_ARCHITECTURE.md` inside for details. |

## Repository Structure

```
clickbait-place/
├── clickbait_motivate/                 # Primary task workflow
├── clickbait_motivate_sleap/           # SLEAP pose-estimation variant
├── clickbait_motivate_sleap_timer/     # Trials time out after (5) seconds;
├── clickbait_odor/                     # Odor-cued variant
├── clickbait_behavior_3port/           # Three-port FMON variant
├── clickbait_dry_morris_maze/          # Virtual Morris maze
├── clickbait_anemo/                    # "Wind cued" variant
├── clickbait_anemo_lognormal/
├── clickbait_master/                   # Place-field recording
├── utils/                              # Hardware configs and test workflows
│   ├── acA2040-90umNIR_24714376.pfs    # Basler camera profile
│   ├── cbp_openephys.config            # OpenEphys 32ch tetrode config
│   ├── normal_distribution_viewer.py   # Distribution visualization utility
│   ├── fan_test/                       # Test Arduino PWM output
│   └── pin_test/                       # Test Arduino digital output
│   └── water_calibration/              # Calibrate water reward volume
│   └── water_flush/                    # Flush tubes with button press
└── wip/                                # Work-in-progress refactors
    └── clickbait-refactor-063025/
```

Each workflow directory contains:
- `<name>.bonsai` — Bonsai-rx workflow (XML)
- `<name>.bonsai.layout` — Visual layout for the Bonsai editor (where applicable)
- `<name>_task.py` — Extracted "Clickbait Task" IronPython node (task variants only)
- `README.md` — Workflow-specific description
