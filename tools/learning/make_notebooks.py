"""Regenerate the output-free learning interface tutorials."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]


def notebook(name, cells):
    book = nbf.v4.new_notebook(cells=[
        nbf.v4.new_markdown_cell(text) if kind == 'md' else nbf.v4.new_code_cell(text)
        for kind, text in cells
    ], metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    nbf.write(book, ROOT / 'notebooks' / name)


setup = '''from pzr import learning
import pandas as pd
import torch

torch.set_num_threads(1)
task = learning.load_task("robot_arm", bundle="starter", budget=40)'''
notebook('00_start_here.ipynb', [
('md', '''# Tutorial: Train and evaluate a reducer selector

This tutorial covers the supervised learning workflow: load recorded data, train a selector, evaluate it, and save a checkpoint. It uses recorded teacher data, so simulation and teacher collection are not required.

Choose **Run → Run All Cells** to execute the complete workflow.

The network scores four sound native reducer actions: `girard`, `scott`, `pca`, `combastel`. Lower scores rank first. The native execution layer handles bounds and fallbacks. Training uses the existing pairwise objective, rather than predicting calibrated costs.

The starter contains budget-40 decision states from training seeds 0–3 and validation seeds 20–21. Validation is for development, not a fresh generalization claim.'''),
('code', setup),
('code', '''pd.DataFrame({"split": ["train", "validation"],
              "states": [task.training.num_samples, task.validation.num_samples]})'''),
('md', '## Train and inspect validation results\nThe tutorial uses twenty epochs to limit runtime. The returned policy uses the best validation checkpoint.'),
('code', '''run = learning.train_selector(task, config=learning.TrainingConfig(epochs=20, seed=42))
pd.Series(learning.evaluate_selector(run.policy, task.validation))'''),
('md', '''## Compare with the frozen reference
The frozen Clean148 model used 148 training trajectories. This is a reference, not a controlled comparison with the four-trajectory starter model. Notebook 01 compares two networks trained on identical data.'''),
('code', '''pd.DataFrame({
    "starter MLP": learning.evaluate_selector(run.policy, task.validation),
    "frozen Clean148 (more training data)": learning.evaluate_selector(task.frozen_baseline(), task.validation),
})'''),
('md', '''## Run in the native monitor
This short prefix checks integration. Prefix losses are explicitly labelled; remove `max_events` for full 500-event validation trajectories. Exact references may take longer than neural training.'''),
('code', '''learning.evaluate_monitor(run.policy, task.validation_traces, max_events=20,
                         reference_cache="workspace/references")'''),
('md', '## Save and reload\nEach execution uses a new directory, so rerunning the notebook preserves earlier experiments.'),
('code', '''from pathlib import Path
from uuid import uuid4
output = Path("workspace") / ("first-selector-" + uuid4().hex[:8])
run.save(output)
reloaded = learning.load_selector(output)
print(output)
assert learning.evaluate_selector(reloaded, task.validation) == learning.evaluate_selector(run.policy, task.validation)'''),
])
notebook('01_build_your_selector.ipynb', [
('md', '''# Tutorial: Customize and compare selector networks
This tutorial covers the model interface, architecture comparison, prediction inspection, and access to the training loss for custom optimization loops.

Define a PyTorch module in `make_network`, then run the remaining cells to compare it with the baseline. The model receives normalized float32 features `[batch, 15]` and returns four finite scores `[batch, 4]`. Normalization is fitted only on training rows. Any PyTorch module satisfying this contract can be used.'''),
('code', setup),
('code', '''from torch import nn

def make_network(input_dim, num_actions):
    return nn.Sequential(
        nn.Linear(input_dim, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, num_actions),
    )'''),
('code', '''config = learning.TrainingConfig(epochs=20, seed=42)
baseline = learning.train_selector(task, config=config)
challenger = learning.train_selector(task, model_factory=make_network, config=config)
pd.DataFrame({
    "baseline": learning.evaluate_selector(baseline.policy, task.validation),
    "my network": learning.evaluate_selector(challenger.policy, task.validation),
})'''),
('md', '''## Inspect predictions
Costs are training/evaluation labels from a two-event terminal-loss teacher. They are never inputs to the policy. The `feasible` mask accompanies NaN costs for unavailable actions. Scores are rankings, not estimated physical costs.'''),
('code', '''sample = 0
pd.DataFrame({"action": task.dataset.candidate_names,
              "score": challenger.policy.predict_scores(task.validation.features[sample]),
              "teacher_cost": task.validation.teacher_costs[sample],
              "feasible": task.validation.feasible[sample]})'''),
('md', '## Compare native trajectories\nEach method sees the same recorded validation inputs. These are prefix diagnostics; use full traces for scientific comparison.'),
('code', '''methods = {"baseline": baseline.policy, "challenger": challenger.policy,
           **{name: name for name in task.dataset.candidate_names}}
reports = {name: learning.evaluate_monitor(policy, task.validation_traces, max_events=20,
                                          reference_cache="workspace/references")
           for name, policy in methods.items()}
pd.concat(reports, names=["method", "row"])[["final_approx_loss", "sum_approx_loss", "fallback_count"]]'''),
('md', '## Custom training loops\nThe same prepared samples and pairwise loss are public. Keep validation out of gradient updates and normalization fitting.'),
('code', '''samples = learning.prepared_samples(task.training, challenger.policy.normalizer)
features, teacher_costs, feasible = next(iter(torch.utils.data.DataLoader(samples, batch_size=32)))
scores = challenger.policy.model(features)
loss = learning.cost_sensitive_pairwise_loss(scores.double(), teacher_costs, feasible)
print("Example batch loss:", float(loss.detach()))'''),
('code', '''from pathlib import Path
from uuid import uuid4
output = Path("workspace") / ("custom-selector-" + uuid4().hex[:8])
challenger.save(output)
loaded = learning.load_selector(output, model_factory=make_network)
print(learning.evaluate_selector(loaded, task.validation))'''),
])
notebook('02_understand_the_monitor.ipynb', [
('md', '''# Tutorial: Inspect monitor specifications, geometry, and actions
This tutorial uses a monitor session to inspect task metadata, pre-event features, zonotope geometry, and action availability.

The Geometry15 policy sees only the current pre-event state. Specification text and channel names describe the task; recorded future inputs and exact references are evaluator-only data.'''),
('code', setup),
('code', '''print(task.metadata["spec"])
print("Inputs:", task.metadata["input_channels"])
print("Triggers:", task.metadata["trigger_labels"])'''),
('code', '''from pzr.rtlola.scenarios import RtlolaTrace
trace = task.validation_traces.traces[0]
prefix = RtlolaTrace(trace.scenario, trace.trace_kind, trace.seed, trace.events[:20])
session = learning.MonitorSession(prefix, budget=task.budget)
for _ in range(5):
    transition = session.step(0)  # request girard
observation = session.observe(geometry=True)
print(transition)
pd.Series(dict(zip(task.dataset.feature_names, observation.features)))'''),
('md', '''## Interpret zonotope geometry
`center[d]` and `generators[d,m]` describe `center + generators @ xi`, with each component of `xi` between -1 and 1. Dynamic and total exports are separate; total includes constant calibration slack.

Signed coefficients and logical zero rows are retained. Do not assume row identities persist across steps, or that total generator columns can be matched to dynamic columns. The compact reducer dimension and exported logical dimension are distinct. The budget is the native transform bound, not a post-event dense-slot limit.'''),
('code', '''print("Dynamic:", observation.dynamic.center.shape, observation.dynamic.generators.shape)
print("Total:", observation.total.center.shape, observation.total.generators.shape)
print("Counts:", observation.counts)
print("Known availability:", observation.action_mask)
print("Reduction needed:", observation.reduction_needed)'''),
('md', '''The arrays are read-only inspection data. Native transforms alone change the monitor. Custom encoders can use these exports now through `learning.FeatureEncoder` (see notebook 03). Recorded teacher bundles contain only Geometry15, so supervised GNN training requires geometry captured at the corresponding teacher states; learned reduction and forecasting require separate targets and soundness checks before any live use.'''),
])
notebook('03_use_gym.ipynb', [
('md', '''# Tutorial: Use the Gymnasium reducer environment
This tutorial covers environment creation, observations, action execution, rewards, and episode boundaries. The Gymnasium adapter supports reinforcement learning experiments with the same native action execution used by supervised selectors. Supervised training can also be used independently of Gym.

One step consumes one recorded event. The observation contains Geometry15, a conservative action mask, and a reduction-needed flag. All actions execute `none` automatically under budget. A requested infeasible action tries the remaining candidates in catalog order, then native `interval` fallback.

The default reward is zero until trace completion, then negative terminal native approximation loss. `objective="cumulative"` instead returns negative loss at every step. Exact references are never in observations or info.'''),
('code', setup),
('code', '''from pzr.learning.gym import ReducerEnv
from pzr.rtlola.scenarios import RtlolaTrace
# A short complete trace for the tutorial, not a full-length scientific result.
traces = learning.TraceSet(tuple(RtlolaTrace(t.scenario, t.trace_kind, t.seed, t.events[:20])
                                for t in task.training_traces.traces[:1]), task.budget, "train")
env = ReducerEnv(traces)
observation, info = env.reset(seed=42)
print(observation)
print(info)'''),
('code', '''env.action_space.seed(42)
rows = []
while True:
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    rows.append({"reward": reward, **info})
    if terminated or truncated:
        break
pd.DataFrame(rows)'''),
('md', '''## Define the model input
`FeatureEncoder` makes the observation-to-tensor computation explicit. The default `learning.GEOMETRY15_ENCODER` returns the raw paper features, whose names are available as `feature_names`. Their implementation is `pzr.rtlola.features.ranking_features_from_matrices`; supervised selectors additionally apply the schema's log transform and train-fitted normalization.

Here we define three alternative properties from the dynamic zonotope. This is a feature-design example, not a validated replacement for Geometry15. The function receives a pre-event observation, with no future inputs or exact reference. Handle the empty initial state and variable geometry sizes explicitly.'''),
('code', '''import numpy as np

def zonotope_properties(observation):
    generators = observation.dynamic.generators
    radius = np.abs(generators).sum(axis=1)
    return np.array([
        observation.budget,
        radius.sum(),
        radius.max(initial=0.0),
    ], dtype=np.float32)

encoder = learning.FeatureEncoder(
    name="dynamic-radius.v1", shape=(3,), function=zonotope_properties,
    requires_geometry=True,
    feature_names=("budget", "radius_sum", "radius_max"),
)
custom_env = ReducerEnv(traces, feature_encoder=encoder)
custom_observation, _ = custom_env.reset(seed=42)
assert custom_env.observation_space.contains(custom_observation)
print(custom_env.observation_space)
print(custom_observation["features"])
# The same function is usable outside Gym.
raw = custom_env.session.observe(geometry=True)
np.testing.assert_array_equal(encoder(raw), custom_observation["features"])'''),
('md', '''## Put a learned embedding inside the policy
A feature function can call a frozen embedding model in evaluation mode. For an embedding that should learn jointly with action selection, return its input from the encoder and put the embedding in your PyTorch policy. Gym returns NumPy arrays, so gradients do not flow through the environment.

This small example keeps the embedding and action head in one differentiable module. For a future GNN, an encoder can instead return a fixed-capacity node/edge tensor with explicit masks; declare that tensor shape and reject capacity overflow rather than silently dropping geometry. Row identities are not stable across events. Graph construction, batching and the GNN remain your model's responsibility.'''),
('code', '''from torch import nn

embedding = nn.Sequential(nn.Linear(3, 8), nn.Tanh())
action_head = nn.Linear(8, 4)
network = nn.Sequential(embedding, action_head)
inputs = torch.from_numpy(custom_observation["features"]).unsqueeze(0)
scores = network(inputs)
assert scores.shape == (1, 4)
# A training loss on scores can update both modules.

def score_one(features):
    network.eval()
    with torch.no_grad():
        return network(torch.from_numpy(features).unsqueeze(0))[0].numpy()

# Native evaluation accepts the same encoder and lower-is-better score order.
policy = learning.ObservationPolicy(encoder, score_one)
learning.evaluate_monitor(policy, traces, max_events=10)
custom_env.close()'''),
('md', '''These weights are untrained; the evaluation only demonstrates integration. An RL optimizer must train the policy using environment transitions. Keep any fitted preprocessing restricted to training data and save it alongside the encoder definition and model weights.

The recorded starter/Clean148 supervised datasets store Geometry15, not full zonotopes. `train_selector` continues to use those recorded features. New geometry-based supervised inputs require collecting geometry at the original teacher decision states; replaying an arbitrary policy does not reconstruct them. An embedding over the existing 15 inputs can already be trained through `model_factory` in notebook 01.'''),
('md', '''## Distinguish truncation and termination
An external limit ends an incomplete episode with `truncated=True`. An incomplete episode receives no terminal reward at the cutoff. Use full trajectories when optimizing the terminal objective.'''),
('code', '''limited = ReducerEnv(traces, max_steps=2)
limited.reset(seed=42)
limited.step(0)
_, reward, terminated, truncated, _ = limited.step(0)
assert truncated and not terminated and reward == 0
limited.close()
env.close()'''),
])
notebook('04_define_zonotope_features.ipynb', [
('md', '''# Tutorial: Define model inputs from zonotope geometry

This tutorial is a hands-on guide to controlling exactly what the policy receives. We will inspect a current zonotope, compute named properties, choose their order and preprocessing, and reuse the resulting encoder in Gym and native policy evaluation.

Run all cells in the learning environment with the pinned native binding. This notebook uses one short training-trace prefix and untrained network weights; it demonstrates the interface, not model quality. Notebook 03 explains actions and rewards; notebook 01 covers supervised training on the recorded Geometry15 data.'''),
('code', setup),
('code', '''import numpy as np
from torch import nn
from pzr.learning.gym import ReducerEnv
from pzr.rtlola.scenarios import RtlolaTrace

trace = task.training_traces.traces[0]
prefix = RtlolaTrace(trace.scenario, trace.trace_kind, trace.seed, trace.events[:20])
traces = learning.TraceSet((prefix,), task.budget, "train")
session = learning.MonitorSession(prefix, budget=task.budget)
for _ in range(5):
    session.step(0)
state = session.observe(geometry=True)'''),
('md', '''## 1. Inspect the inputs available to a feature function

A zonotope is `c + G @ xi`, with `xi` in `[-1, 1]^m`: `center` has shape `[d]` and `generators` has shape `[d, m]`. Its coordinate radius is `r[i] = sum_j abs(G[i, j])`; coordinate widths are `2*r`. These properties describe the current pre-event state, without a future rollout.

`dynamic` excludes constant calibration slack; `total` includes it. Exported logical zero rows and signed coefficients are preserved. Do not treat the exported shape as the compact reducer dimension, assume stable row identities across events, or match dynamic and total columns. Arrays are read-only; only native reducers modify monitor state.'''),
('code', '''for name in ("dynamic", "total"):
    geometry = getattr(state, name)
    print(name, "center:", geometry.center.shape, "generators:", geometry.generators.shape)
print("Native counts:", state.counts)
print("Budget:", state.budget)

# The default encoder exposes raw Geometry15, even when custom inputs are used.
pd.Series(learning.GEOMETRY15_ENCODER(state),
          index=learning.GEOMETRY15_ENCODER.feature_names, name="raw Geometry15")'''),
('md', '''## 2. Define a small catalogue of properties

The function below returns scalar properties with explicit names. Empty initial geometry produces zero radii and zero center magnitude. Absolute values preserve the contribution of negative coefficients. Radius sums and maxima are invariant to reordering coordinates or generators; this avoids relying on persistent row positions.

These are illustrative features, not the paper's complete Geometry15 computation. Width is an input property here, not a replacement for the native approximation-loss objective.'''),
('code', '''def zonotope_properties(observation):
    dynamic = observation.dynamic
    total = observation.total
    dynamic_radius = np.abs(dynamic.generators).sum(axis=1)
    total_radius = np.abs(total.generators).sum(axis=1)
    return {
        "budget": float(observation.budget),
        "dynamic_radius_sum": float(dynamic_radius.sum()),
        "dynamic_radius_max": float(dynamic_radius.max(initial=0.0)),
        "dynamic_center_abs_max": float(np.abs(dynamic.center).max(initial=0.0)),
        "total_radius_sum": float(total_radius.sum()),
    }

pd.Series(zonotope_properties(state), name="property value")'''),
('md', '''## 3. Choose the tensor's contents, order, and scale

**Edit `selected_features` to add, remove, or reorder inputs.** Set `use_log1p=False` for raw values. This example's properties are all nonnegative, so `log1p(x)` is defined and compresses their scale. Signed features need a different transformation. If you later fit a normalizer, fit it on training states only and reuse it unchanged for validation and inference.

Rerun this cell and the remaining cells after editing. The declared shape, feature names, Gym space, and network input dimension then agree. Change the encoder name/version whenever its definition changes.'''),
('code', '''selected_features = (
    "budget",
    "dynamic_radius_sum",
    "dynamic_radius_max",
    "total_radius_sum",
)
use_log1p = True

# Capture the configuration so later variable edits cannot silently change this encoder.
def make_feature_function(names, log_scale):
    names = tuple(names)

    def encode(observation):
        properties = zonotope_properties(observation)
        values = np.array([properties[name] for name in names], dtype=np.float64)
        if log_scale:
            if np.any(values < 0):
                raise ValueError("This log1p example requires nonnegative properties")
            values = np.log1p(values)
        return values.astype(np.float32)

    return encode

encoder = learning.FeatureEncoder(
    name="tutorial-zonotope-properties.v1",
    shape=(len(selected_features),),
    function=make_feature_function(selected_features, use_log1p),
    requires_geometry=True,
    feature_names=selected_features,
)
pd.DataFrame({
    "raw property": [zonotope_properties(state)[name] for name in encoder.feature_names],
    "model input": encoder(state),
}, index=encoder.feature_names)'''),
('md', '''`FeatureEncoder` checks the declared shape and rejects NaN or infinite outputs. `requires_geometry=True` asks Gym to export geometry for the function. Without Gym, call `session.observe(geometry=True)` yourself. The original `Observation.features` remains Geometry15; the custom vector becomes Gym's `observation["features"]`.'''),
('code', '''# A simple known-answer check for the radius calculation, including negative entries.
example_generators = np.array([[1.0, -2.0], [0.0, 0.0]])
np.testing.assert_array_equal(np.abs(example_generators).sum(axis=1), [3.0, 0.0])
assert encoder(state).shape == encoder.shape
assert encoder(state).dtype == np.float32
assert np.isfinite(encoder(state)).all()'''),
('md', '''## 4. Inspect the actual Gym inputs over time

Reset tests the empty initial state. The table records observations **before** each event, which is when the policy chooses an action. We request Girard throughout to make this demonstration deterministic. Changing features does not change the native action catalogue, transform bounds, fallback rules, or reward definition.'''),
('code', '''env = ReducerEnv(traces, feature_encoder=encoder)
observation, _ = env.reset(seed=42, options={"trace_index": 0})
print(env.observation_space)
rows = []
while True:
    assert env.observation_space.contains(observation)
    native_observation = env.session.observe(geometry=True)
    np.testing.assert_array_equal(observation["features"], encoder(native_observation))
    rows.append(dict(zip(encoder.feature_names, observation["features"])))
    observation, reward, terminated, truncated, info = env.step(0)
    if terminated or truncated:
        break
env.close()
pd.DataFrame(rows).rename_axis("pre-event index")'''),
('md', '''## 5. Feed the vector into a learned representation and action head

Keep trainable embeddings inside the PyTorch policy: gradients cannot pass through Gym's NumPy observations. Here a small embedding maps `[batch, num_features]` to `[batch, 8]`; the action head returns `[batch, 4]`. Both modules can be optimized jointly by your training loop. The weights below are seeded but **untrained**.'''),
('code', '''with torch.random.fork_rng():
    torch.manual_seed(42)
    network = nn.Sequential(
        nn.Linear(encoder.shape[0], 8), nn.Tanh(),  # learned representation
        nn.Linear(8, 4),                         # action scores
    )
inputs = torch.from_numpy(encoder(state)).unsqueeze(0)
scores = network(inputs)
assert scores.shape == (1, 4)
print("Input shape:", tuple(inputs.shape), "score shape:", tuple(scores.shape))

def score_one(features):
    network.eval()
    with torch.no_grad():
        return network(torch.from_numpy(features).unsqueeze(0))[0].numpy()

policy = learning.ObservationPolicy(encoder, score_one)
learning.evaluate_monitor(policy, traces, max_events=10)'''),
('md', '''The same encoder now drives native evaluation. Scores are lower-is-better, ordered `girard`, `scott`, `pca`, `combastel`. This integration check runs only ten events; it does not train the network or establish performance.

## Next experiments

- Add a property inside `zonotope_properties`, include its name in `selected_features`, and rerun from section 3. Keep the empty-state and finite-value checks.
- For a frozen embedding, the feature function can call a model in evaluation mode and return its fixed-size output. For jointly learned representations, keep the embedding inside the policy as above.
- A GNN needs geometry-preserving inputs rather than these aggregates. `FeatureEncoder` also accepts multidimensional fixed shapes, so a future graph encoder can pack padded nodes/edges and masks. Reject capacity overflow instead of dropping generators; do not use unstable row identities as persistent node IDs. Graph construction and batching require an additional implementation.

The starter and Clean148 bundles contain **Geometry15 only**, not the original zonotope matrices. This notebook does not alter `train_selector`'s recorded inputs. You can train an embedding over those 15 features using notebook 01's `model_factory`. Training supervised selectors on new geometric properties requires geometry captured at the matching teacher decision states; arbitrary replay cannot reconstruct those states or justify reusing their labels.

Save the feature function, selected names/order, transform settings, encoder version, any fitted preprocessing, and network weights together. Custom encoders are not serialized by `TrainingRun.save`. No live zonotope prediction or matrix writeback is introduced here.'''),
])
