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
('md', '''The arrays are read-only inspection data. Native transforms alone change the monitor. Future GNN datasets will use these geometry exports; learned reduction and forecasting require separate targets and soundness checks before any live use.'''),
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
