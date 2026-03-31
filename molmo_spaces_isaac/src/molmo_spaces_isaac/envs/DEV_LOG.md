# Isaac Envs — Development Log

## 2026-03-31: Stack variants + gate improvements

### Completed
- **FrankaStackEnv** — 3 task variants via `stack_mode` config:
  - `homogeneous`: target is same category as stack (e.g. bottom bowl in stack of bowls)
  - `container_target`: different container at bottom (e.g. cup under plates)
  - `flat_target`: flat object at bottom (e.g. cellphone under plates)
- **Stability gate check** — position + velocity (linear + angular) checks after 30 settle steps
- **FrankaTransferEnv** — gate check (food on source), food size filter (must fit in container), unique seed per reset
- **FrankaClutterEnv** — pool-based sampling (16 objects, 8 active per episode), per-env diversity, continuous gate check, ring-based packing
- **Tests moved** to `envs/tests/` folder, `--mode` arg for stack test

## 2026-03-30: Initial task env pipeline

### Completed
- **AssetRegistry** — discovers THOR USD assets from `~/.molmospaces`, filters by bbox for tabletop suitability (417 assets, 62 categories)
- **FrankaTableEnv base class** — shared robot/table/ground/lighting/action handling
- **FrankaPickupEnv** — refactored to inherit from base
- **FrankaClutterEnv, FrankaStackEnv, FrankaTransferEnv, FrankaPinchEnv** — initial implementations
- **test scripts** — smoke tests with video recording (timestamped output dirs)

### Open items

#### FrankaClutterEnv
- [ ] **Robot edge alignment**: place robot at table edge facing target. Low priority.
- [ ] **Expose active object roles**: needed before adding rewards/policies.
- [ ] **Richer observation space**: encode fragile/clutter positions for safety-aware policies.
- [ ] **Episode diagnostics logging**: JSONL with pack attempts, gate results, object selections.

#### FrankaPinchEnv
- [ ] Apply pool-based sampling + gate check
- [ ] Validate pinch gap is maintained after settling

#### AssetRegistry
- [ ] **Investigate raising max_dim threshold for more objects**: currently defaults to 0.25m (417 of 1335 non-articulated assets). Raising to 0.30m gives 515, 0.50m gives 696. Many larger objects (lamps, TVs, dog beds) aren't sensible for tabletop manipulation, so this needs per-category curation rather than a blanket threshold increase. Each task env can already override `max_object_dim` independently.

#### General
- [ ] Rewards / success criteria for all task envs
- [ ] Cabinet-based tasks (blocked door, cabinet clutter) — requires articulated furniture
- [ ] Liquid transport — requires particle sim not available in IsaacLab
