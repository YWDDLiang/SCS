# Stage-Dependent Data Scheduling for Generative Recommendation

Anonymous code release accompanying the manuscript *What to Learn and How to
Learn: Stage-Dependent Data Scheduling for Generative Recommendation*.

The implementation uses TIGER as its recommendation backbone. The
Spectrum-Conflict Scheduler (SCS) coordinates two decisions during training:
which interactions to prioritize as representations evolve, and whether each
additional presentation uses the original or an alternative history.

## Method

At each stage, SCS combines online representation-spectrum novelty with a
cross-user coverage-qualified event pool. Events that admit an alternative
history receive higher mass within the same priority distribution. The
gradient-conflict signal then guides the allocation of history views within a
fixed view quota. A view preserves the target, history length, latest history
item, and temporal order.

The reported configuration uses a minimum transition coverage of two distinct
users in the original training split. The full base training stream remains
active throughout training. See [METHOD.md](METHOD.md) for the event-priority
definition and complete scheduling settings.

## Repository Layout

- `src/data/loading/datamodules/spectrum_conflict_scheduler.py` implements
  online event scheduling and history construction.
- `src/data/selection/spectrum_event_priority.py` contains the combined
  spectrum and transition-coverage distribution.
- `src/data/selection/representation_spectrum_scheduler.py` computes online
  representation-spectrum priorities.
- `src/data/selection/conflict_aware_presentation.py` maps gradient conflict
  to history-view allocation probabilities.
- `scripts/run_spectrum_scheduled_views.py` is the four-GPU training entry
  point.
- `tests/` contains focused tests for the scheduling rules and policy.

## Setup

Use Python 3.10 and install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

Prepare the public Amazon benchmark data in the format used by this
implementation. Training requires the original training candidate store, a
completed training-occurrence trace, cached validation and test batches, and the
immutable replay assets derived from that trace. Pass those paths explicitly
to the runner; set `REPLAY_SHUFFLE_ASSETS` to the replay-assets directory.
The runner does not download data or pretrained assets.

## Training

The default settings match the reported SCS configuration: seed 42, 200-update
warm-up, 32-dimensional spectrum projection, 0.99 covariance decay, 0.01
ridge, 0.90 conflict-reference decay, a 4x view-readiness weight, a 75%
history-view quota, and domain-specific additional-presentation budgets.

```bash
REPLAY_SHUFFLE_ASSETS=/path/to/replay-assets \
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/run_spectrum_scheduled_views.py \
  --dataset toys \
  --mode native_replay \
  --output-dir outputs/toys_scs_seed42 \
  --candidate-store /path/to/toys_train_candidates.pt \
  --source-trace-dir /path/to/toys_training_trace \
  --eval-cache-dir /path/to/evaluation-cache \
  --fast-data \
  --loader-context spawn \
  --loader-timeout 600
```

Replace `toys` with `beauty` or `sports` and provide the corresponding
training and evaluation assets. The output contains the composed configuration,
run metadata, online scheduling trace, checkpoints, and final metrics.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

The accompanying license and third-party notices are retained.
