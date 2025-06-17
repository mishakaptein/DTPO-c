from gymnax.environments.classic_control.pendulum import Pendulum, EnvParams, EnvState

from gymnax.environments import spaces

import jax

from typing import Tuple, Optional

import chex


class PendulumBangBang(Pendulum):
    def __init__(self, num_actions: int = 2):
        super().__init__()

        self._num_actions = num_actions
        max_torque = self.default_params.max_torque
        self.action_values = jax.numpy.linspace(-max_torque, max_torque, num_actions)
        self.feature_names = ["cos theta", "sin theta", "theta dot"]
        # Here we represent the actual used torque values for a clearer interpretation (instead of an idx):
        self.action_names = ["torque_" + str(i) for i in self.action_values]

    @property
    def name(self) -> str:
        """Environment name."""
        return "Pendulum-v1-BangBang"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self._num_actions

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Discrete(self._num_actions)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: float,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        action_space = super().action_space()
        continuous_action = jax.numpy.array([self.action_values[action]])
        return super().step_env(key, state, continuous_action, params)
