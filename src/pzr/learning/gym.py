"""Optional Gymnasium adapter; importing pzr.learning does not require Gym."""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from pzr.learning.encoding import FeatureEncoder, GEOMETRY15_ENCODER
from pzr.learning.monitor import MonitorSession
from pzr.learning.tasks import TraceSet


class ReducerEnv(gym.Env):
    """One event per step; configurable fixed-shape pre-event feature tensors.

    ``feature_encoder`` defines the features Box; masks and native execution
    are independent of the encoding. Reward references never enter observations
    or info. See FeatureEncoder for geometry and learned-embedding contracts.
    """
    metadata = {'render_modes': []}

    def __init__(self, traces: TraceSet, *, objective: str = 'terminal', max_steps: int | None = None,
                 feature_encoder: FeatureEncoder = GEOMETRY15_ENCODER):
        if objective not in ('terminal', 'cumulative'):
            raise ValueError('objective must be terminal or cumulative')
        if not traces.traces or (max_steps is not None and max_steps < 1):
            raise ValueError('nonempty traces and positive max_steps required')
        self.traces, self.objective, self.max_steps = traces, objective, max_steps
        self.feature_encoder = feature_encoder
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Dict({
            'features': gym.spaces.Box(-np.inf, np.inf, feature_encoder.shape, np.float32),
            'action_mask': gym.spaces.MultiBinary(4), 'reduction_needed': gym.spaces.Discrete(2),
        })
        self.session: MonitorSession | None = None
        self._ended = False

    def _observation(self):
        obs = self.session.observe(geometry=self.feature_encoder.requires_geometry)
        return {'features': self.feature_encoder(obs), 'action_mask': obs.action_mask.copy(),
                'reduction_needed': int(obs.reduction_needed)}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        index = (options or {}).get('trace_index')
        if index is None:
            index = int(self.np_random.integers(len(self.traces.traces)))
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self.traces.traces):
            raise ValueError('trace_index outside supplied trace set')
        self.session = MonitorSession(self.traces.traces[index], budget=self.traces.budget)
        self._ended = False
        return self._observation(), {'trace_index': int(index)}

    def step(self, action):
        if self.session is None or self._ended:
            raise RuntimeError('call reset before stepping a new episode')
        if not self.action_space.contains(action):
            raise ValueError('action outside Discrete(4)')
        record = self.session.step(int(action))
        terminated = self.session.done
        truncated = not terminated and self.max_steps is not None and self.session.index >= self.max_steps
        self._ended = terminated or truncated
        reward = -record['approx_loss'] if self.objective == 'cumulative' or terminated else 0.0
        info = {k: record[k] for k in ('requested_action', 'executed_action', 'fallback_used',
                                     'infeasible_candidate_count', 'override_reason')}
        return self._observation(), float(reward), terminated, truncated, info

    def close(self):
        self.session = None
