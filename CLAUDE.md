# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MolmoSpaces is a large-scale robotics simulation framework for robot manipulation and navigation. It supports multiple simulators (MuJoCo, IsaacSim, ManiSkill), provides asset management, benchmark evaluation, teleoperation, and data generation pipelines.

## Build & Development Commands

```bash
# Install (uses uv package manager, Python 3.10)
uv pip install -e .

# Install dev dependencies
uv pip install -e . --group dev

# Install pre-commit hooks
pre-commit install

# Lint and format
ruff check .
ruff format .

# Run pre-commit on staged files
pre-commit run

# Data generation example
export PYTHONPATH=${PYTHONPATH}:.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python -m molmo_spaces.data_generation.main \
    molmo_spaces/data_generation/config/<config_file>.py \
    <ConfigClassName>

# Benchmark evaluation
python -m molmo_spaces.evaluation.eval_main <eval_config>

# House generation
generate-houses  # entry point script

# Tests (directory: mlspaces_tests/)
PYTHONPATH=. pytest mlspaces_tests/data_generation
PYTHONPATH=. pytest mlspaces_tests/data_generation --log-cli-level DEBUG
```

## Code Style

- **Formatter/Linter**: Ruff (line-length 100, double quotes, Python 3.10 target)
- **Docstrings**: Google convention
- **Type checker**: ty (with extra paths in `./typings`)

## Architecture

### Three Packages

- **`molmo_spaces/`** — Main package (MuJoCo-based simulation, data generation, evaluation)
- **`molmo_spaces_isaac/`** — IsaacSim/IsaacLab integration (asset conversion MJCF→USD, separate pyproject.toml, Python >=3.11)
- **`molmo_spaces_maniskill/`** — ManiSkill/Sapien integration (MJCF loaders, separate pyproject.toml)

### Config-Driven Design

Hierarchical Pydantic configuration: `ExperimentConfig` → `TaskSamplerConfig` → `TaskConfig` / `RobotConfig` / `PolicyConfig`. Config classes auto-register via a registry pattern in `data_generation/config_registry.py`. The entry point `data_generation/main.py` auto-discovers config files.

### Core Abstractions (in `molmo_spaces/`)

| Layer | Base Class | Location |
|-------|-----------|----------|
| Robot | `Robot` (abstract) | `robots/abstract.py` |
| Task | `Task` / `TaskSampler` | `tasks/task.py`, `tasks/task_sampler.py` |
| Policy | `InferencePolicy` | `policy/base_policy.py` |
| Controller | Abstract controller | `controllers/abstract.py` |
| Environment | MuJoCo wrapper | `env/env.py` |
| Planner | Abstract planner | `planner/abstract.py` |

### Data Generation Pipeline Flow

```
ExperimentConfig → TaskSampler (scene randomization) → Environment (MuJoCo)
→ Policy (action generation) → Robot Controller → Simulation Step
→ Observation (cameras, proprioception) → Task Success Check → H5 Save
```

Parallel execution via `ParallelRolloutRunner` in `data_generation/pipeline.py`.

### Data Format

Trajectories are stored as HDF5 files with structure: `traj_{ep_idx}/` containing `actions/`, `obs/`, `rewards`, `success`, `terminated`, `truncated`.

### Key Constants

`molmo_spaces/molmo_spaces_constants.py` — Central registry for asset paths, data versions, and environment variable overrides:
- `MLSPACES_ASSETS_DIR` — Asset installation directory
- `MLSPACES_CACHE_DIR` — Resource cache (default: `~/.cache/molmo-spaces-resources`)
- `MLSPACES_AUTO_INSTALL` — Auto-install assets without prompting
- `MLSPACES_PINNED_ASSETS_FILE` — Override asset versions via JSON

### Evaluation

JSON-based benchmark evaluation in `evaluation/`. Entry point: `eval_main.py`. Benchmark schemas define episodes with poses, cameras, and tasks. Results aggregate to W&B.

### Robots

Implementations: Franka (`robots/franka.py`), RBY1 (`robots/rby1.py`), floating base variants. Each has corresponding kinematics in `kinematics/`.

### External Dependencies

- Git submodule: `external_src/Manifold` (mesh processing)
- `bpy==3.6.0` must be installed separately from Blender's PyPI index before other deps

## Git & Collaboration Workflow

- **Commit frequently**: when a logical unit of work is complete (new feature, bug fix, refactor), propose a commit. Do not batch unrelated changes.
- **Always get approval first**: before committing, show the user the proposed commit message and list of files. Do not commit without explicit approval.
- **Commit message style**: `type(scope): short description` (e.g. `feat(isaac):`, `fix(clutter):`, `docs:`, `chore:`). Body explains "why", not "what".
- **After each commit**: add a one-line summary to the relevant `DEV_LOG.md` recording what was accomplished.
- **Separate concerns**: reference code, documentation, and feature code go in separate commits.
- **Never force-push** to shared branches without asking.
