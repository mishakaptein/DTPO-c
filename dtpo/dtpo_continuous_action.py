import json
import os
import time

from functools import partial

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._tree import Tree
from sklearn.utils import check_random_state

import numpy as np

import jax
import jax.numpy as jnp
from jax import grad, jit

from gymnax.environments import spaces
from gymnax.environments.environment import Environment

import gymnasium

import rlax

import optax

import flax.linen as nn
from flax.training.train_state import TrainState

from tqdm import tqdm


class ParallelEnvRollouts:
    def __init__(self, env, num_envs, n_steps, gamma, gae_lambda):
        self.env = env
        self.num_envs = num_envs
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.buffer_size = num_envs * n_steps

    def reset(self, rng):
        env_rngs = jax.random.split(rng, self.num_envs)
        return jax.vmap(self.env.reset, in_axes=(0, None))(
            env_rngs, self.env.default_params
        )

    @partial(jax.jit, static_argnums=(0, 1))
    def get_batch(self, tree_apply, tree_params, curr_observations, curr_states, rng):
        def policy_step(state_input, tmp):
            """lax.scan compatible step transition in jax env."""
            obs, state, rng = state_input
            rng, rng_step, rng_tree = jax.random.split(rng, 3)
            out = jax.vmap(tree_apply, in_axes=(None, 0))(tree_params, obs)
            mu, log_sigma = jnp.split(out, 2, axis=1)
            sigma = jnp.exp(log_sigma)
            rng, rng_eps = jax.random.split(rng_tree)
            eps = jax.random.normal(rng_eps, shape=mu.shape)
            action = mu + sigma * eps
            action = jnp.clip(action, -2.0, 2.0)

            next_obs, next_state, reward, done, _ = jax.vmap(
                self.env.step, in_axes=(0, 0, 0, None)
            )(
                jax.random.split(rng_step, self.num_envs),
                state,
                action,
                self.env.default_params,
            )
            carry = [
                next_obs,
                next_state,
                rng,
            ]
            y = [obs, action, reward, done]
            return carry, y

        # Scan over episode step loop
        carry, scan_out = jax.lax.scan(
            policy_step,
            [
                curr_observations,
                curr_states,
                rng,
            ],
            (),
            self.n_steps,
        )

        new_observations, new_states, _ = carry
        obs, action, reward, done = scan_out
        return obs, action, reward, done, new_observations, new_states


def random_rollout_gymnax(env, env_params, rng_input, n_steps):
    # Reset the environment
    rng_reset, rng_episode = jax.random.split(rng_input)
    obs, state = env.reset(rng_reset, env_params)

    def policy_step(state_input, tmp):
        """lax.scan compatible step transition in jax env."""
        obs, state, rng = state_input
        rng, rng_step, rng_action = jax.random.split(rng, 3)
        action = jax.random.choice(rng_action, env.num_actions)
        next_obs, next_state, reward, done, _ = env.step(
            rng_step, state, action, env_params
        )
        carry = [
            next_obs,
            next_state,
            rng,
        ]
        y = [obs, action, reward, done]
        return carry, y

    # Scan over episode step loop
    _, scan_out = jax.lax.scan(
        policy_step,
        [
            obs,
            state,
            rng_episode,
        ],
        (),
        n_steps,
    )

    obs, action, reward, done = scan_out
    return obs, action, reward, done


def get_discretized_tree_continuous(tree_params, n_features_in, prune=True):
    """
    Returns a scikit-learn Tree object with the pruned and
    discretized decision tree policy.
    """

    # We want to have 2 outputs per leaf (mean and standard deviation)
    n_leaf_outputs = tree_params["params"]["leaf_params"].shape[1]
    classes_per_output = np.ones(n_leaf_outputs, dtype=int)
    tree = Tree(n_features_in, classes_per_output, n_leaf_outputs)

    tree_params = tree_params.copy()

    def _prune(node_id=0):
        left_id = tree_params["params"]["children_left"][node_id]
        right_id = tree_params["params"]["children_right"][node_id]

        if left_id >= 0:
            _prune(left_id)
            _prune(right_id)
        else:
            pass

    if prune:
        _prune()

    node_count = len(tree_params["params"]["features"])
    nodes = np.zeros(node_count, dtype=[
       ("left_child", np.int64),
       ("right_child", np.int64),
       ("feature", np.int64),
       ("threshold", np.float64),
       ("impurity", np.float64),
       ("n_node_samples", np.int64),
       ("weighted_n_node_samples", np.float64),
    ])
    for i in range(node_count):
       nodes[i] = (
           tree_params["params"]["children_left"][i],
           tree_params["params"]["children_right"][i],
           tree_params["params"]["features"][i],
           float(tree_params["params"]["thresholds"][i]),
           0.0,
           0,
           0.0,
       )

    values = []
    for i in range(node_count):
        mu, log_sigma = tree_params["params"]["leaf_params"][i]
        sigma = float(np.exp(log_sigma))
        values.append(np.array([[mu, sigma]]).reshape(1, n_leaf_outputs, 1))

    values = np.concatenate(values, axis=0).astype(np.float64)
    state = {
       "max_depth": int(tree_params.get("max_depth", -1)),
       "node_count": node_count,
       "nodes": nodes,
       "values": values,
       "n_features_": n_features_in,
       "n_outputs_": n_leaf_outputs,
       "n_classes_": classes_per_output,
    }

    tree.__setstate__(state)
    return tree



class ValueNetwork(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x


class ValueFunctionLinear(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(1)(x)
        return x


class DecisionTreePolicy(nn.Module):
    n_actions: int
    max_nodes: int

    @nn.compact
    def __call__(self, obs: jnp.ndarray):
        features = self.param(
            "features", nn.initializers.constant(0, int), self.max_nodes
        )
        thresholds = self.param(
            "thresholds", nn.initializers.constant(0, float), self.max_nodes
        )
        children_left = self.param(
            "children_left", nn.initializers.constant(-1, int), self.max_nodes
        )
        children_right = self.param(
            "children_right", nn.initializers.constant(-1, int), self.max_nodes
        )
        leaf_params = self.param(
            "leaf_params",
            nn.initializers.constant(0, float),
            (self.max_nodes, self.n_actions),
        )

        def cond(node_id):
            return children_left[node_id] != children_right[node_id]

        def body(node_id):
            return jax.lax.select(
                obs[features[node_id]] <= thresholds[node_id],
                children_left[node_id],
                children_right[node_id],
            )

        leaf_node_id = jax.lax.while_loop(cond, body, 0)

        return leaf_params[leaf_node_id]


@partial(jax.jit, static_argnums=(0, 2, 4))
def deterministic_rollouts(tree_apply, tree_params, n_steps, rng, env):
    # Reset the environment
    rng_reset, rng_episode = jax.random.split(rng)
    obs, state = env.reset(rng_reset, env.default_params)

    def policy_step(state_input, tmp):
        """lax.scan compatible step transition in jax env."""
        obs, state, rng = state_input
        rng, rng_step = jax.random.split(rng)
        out = tree_apply(tree_params, obs)
        mu, _ = jnp.split(out, 2)
        action = mu

        next_obs, next_state, reward, done, _ = env.step(
            rng_step, state, action, env.default_params
        )
        carry = [
            next_obs,
            next_state,
            rng,
        ]
        y = [obs, reward, done]
        return carry, y

    # Scan over episode step loop
    _, scan_out = jax.lax.scan(
        policy_step,
        [
            obs,
            state,
            rng_episode,
        ],
        (),
        n_steps,
    )

    observations, rewards, done = scan_out

    return observations, rewards, done


def clipped_value_loss(
    params, apply_network, pred_values, observations, targets, ppo_epsilon
):
    """
    Clipped PPO loss for the value function.
    """
    new_values = apply_network(params, observations).ravel()
    v_loss_unclipped = (new_values - targets) ** 2
    v_clipped = pred_values + jnp.clip(
        new_values - pred_values,
        -ppo_epsilon,
        ppo_epsilon,
    )
    v_loss_clipped = (v_clipped - targets) ** 2
    v_loss_max = jnp.maximum(v_loss_unclipped, v_loss_clipped)
    return 0.5 * v_loss_max.mean()


@jit
def ppo_loss_samples_continuous(
    logits,
    prev_logp,
    advantage,
    actions,
    epsilon,
):
    """
    PPO loss for the policy.
    """
    # Here we compute the new log probability from N(mu, sigma)
    mu, log_sigma = jnp.split(logits, 2, axis=1)
    sigma = jnp.exp(log_sigma)
    logp_new = jnp.sum(
        -0.5 * (
            ((actions - mu) / sigma) ** 2 + 2 * log_sigma + jnp.log(2 * jnp.pi)
        ),
        axis = 1,
    )
    ratio = jnp.exp(logp_new - prev_logp + 1e-8)
    clipped = jnp.clip(ratio, 1 - epsilon, 1 + epsilon) * advantage
    return jnp.sum(jnp.minimum(ratio * advantage, clipped))


@partial(jax.jit, static_argnums=(0,))
def update_batch(
    apply_network,
    value_opt_state,
    batch_indices,
    pred_values,
    observations,
    target_values,
    ppo_epsilon,
):
    """
    Update the value function with PPO's clipped value loss.
    """
    grads = grad(clipped_value_loss)(
        value_opt_state.params,
        apply_network,
        pred_values[batch_indices],
        observations[batch_indices],
        target_values[batch_indices],
        ppo_epsilon,
    )
    value_opt_state = value_opt_state.apply_gradients(grads=grads)

    return value_opt_state


def sklearn_tree_to_tree_params_continuous(tree, max_nodes):
    """
    Turns a DecisionTreeRegressor into a tree params for optimization
    """
    features = jnp.pad(
        tree.feature, (0, max_nodes - tree.node_count),
        constant_values=-1
    )
    thresholds = jnp.pad(
        tree.threshold, (0, max_nodes - tree.node_count),
        constant_values=-1.0
    )
    children_left = jnp.pad(
        tree.children_left, (0, max_nodes - tree.node_count),
        constant_values=-1
    )
    children_right = jnp.pad(
        tree.children_right, (0, max_nodes - tree.node_count),
        constant_values=-1
    )
    leaf_vals = jnp.squeeze(tree.value, axis=2)
    leaf_params = jnp.pad(
        leaf_vals,
        [(0, max_nodes - leaf_vals.shape[0]), (0, 0)],
        constant_values=0.0
    )
    return {"params": {
        "features": features,
        "thresholds": thresholds,
        "children_left": children_left,
        "children_right": children_right,
        "leaf_params": leaf_params,
    }}


def sklearn_tree_to_string(tree):
    """
    Turn the tree in sklearn format into a nice string representation.
    """
    feature = tree.feature
    threshold = tree.threshold
    children_left = tree.children_left
    children_right = tree.children_right
    value = tree.value
    node_frequency = tree.n_node_samples / tree.n_node_samples[0]

    def tree_to_string(node_id=0, depth=0):
        padding = "  " * depth

        # if this is a leaf
        if children_left[node_id] == children_right[node_id]:
            return f"{padding}{np.round(value[node_id].ravel(), 2)} ({round(node_frequency[node_id] * 100)}%)"

        # if this is a node
        left_subtree = tree_to_string(children_left[node_id], depth + 1)
        right_subtree = tree_to_string(children_right[node_id], depth + 1)

        return f"{padding}if obs[{feature[node_id]}] <= {threshold[node_id]}:\n{left_subtree}\n{padding}else:\n{right_subtree}"

    return tree_to_string()


def learn_tree_structure_with_grad_batch(
    observations,
    advantage,
    actions,
    prev_logp,
    tree_apply,
    tree_params,
    max_depth,
    max_leaf_nodes,
    random_state,
    epsilon,
    learn_rate,
    max_nodes,
    frac,
    grad_clip_norm,
):
    """
    Learn a new decision tree with one gradient update of PPO
    """
    curr_logits = jax.vmap(tree_apply, in_axes=[None, 0])(tree_params, observations)

    gradients = grad(ppo_loss_samples_continuous)(
        curr_logits,
        prev_logp,
        advantage,
        actions,
        epsilon,
    )

    # Clip the global L2 norm
    max_norm = grad_clip_norm
    g_norm = jnp.linalg.norm(gradients)
    clip_coef = jnp.minimum(1.0, max_norm / (g_norm + 1e-6))
    gradients = gradients * clip_coef

    raw_targets = curr_logits + learn_rate * gradients

    # We make sure to replace any NaN or inf with 0
    new_targets = np.nan_to_num(
        np.array(jax.device_get(raw_targets)),
        nan = 0.0, posinf = 0.0, neginf = 0.0
    )

    new_tree = DecisionTreeRegressor(
        max_depth = max_depth,
        max_leaf_nodes = max_leaf_nodes,
        random_state = random_state,
    )

    # Make sure the values stay reasonable by clipping them
    new_targets[:, 0] = np.clip(new_targets[:, 0], -2.0, 2.0)
    new_targets[:, 1] = np.clip(new_targets[:, 1], -5.0, +2.0)

    new_tree.fit(observations, new_targets)

    new_tree_params = sklearn_tree_to_tree_params_continuous(new_tree.tree_, max_nodes)

    # Linearly decay sigma from 1.0 to 0.1 (starting from 75 percent of training)
    start_decay = 0.75
    if frac <= start_decay:
        curr_sigma = 1.0
    else:
        t = (frac - start_decay) / (1.0 - start_decay)
        curr_sigma = 1.0 + (0.1 - 1.0) * t
    log_sigma_val = jnp.log(curr_sigma)
    new_tree_params["params"]["leaf_params"] = (
        new_tree_params["params"]["leaf_params"]
        .at[:, 1].set(log_sigma_val)
    )

    new_logits = jax.vmap(tree_apply, in_axes=[None, 0])(new_tree_params, observations)
    loss_after = ppo_loss_samples_continuous(
        new_logits,
        prev_logp,
        advantage,
        actions,
        epsilon,
    )

    return new_tree_params, loss_after


class DTPOLearner:
    def __init__(
        self,
        env,
        rng,
        ppo_epsilon=0.2,
        gamma=0.99,
        normalize_advantage=True,
        simulation_steps=10000,
        num_envs=1,
        max_iterations=1500,
        criterion="squared_error",
        max_depth=None,
        max_leaf_nodes=16,
        learning_rate=1.0,
        anneal_lr=False,
        max_policy_updates=1,
        early_stop_entropy=0.01,
        cache_clear_interval=10000,
        warmup_iterations=0,
        use_linear_value_function=False,
        verbose=False,
        logging=True,
        random_state=None,
        grad_clip_norm=100.0,
    ):
        self.env = env
        self.rng = rng
        self.ppo_epsilon = ppo_epsilon
        self.gamma = gamma
        self.normalize_advantage = normalize_advantage
        self.simulation_steps = simulation_steps
        self.num_envs = num_envs
        self.max_iterations = max_iterations
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.learning_rate = learning_rate
        self.anneal_lr = anneal_lr
        self.max_policy_updates = max_policy_updates
        self.early_stop_entropy = early_stop_entropy
        self.cache_clear_interval = cache_clear_interval
        self.warmup_iterations = warmup_iterations
        self.use_linear_value_function = use_linear_value_function
        self.verbose = verbose
        self.logging = logging
        self.random_state = random_state
        self.grad_clip_norm = grad_clip_norm

        self.random_state_ = check_random_state(self.random_state)

        # Check if the environment has discrete actions and observations
        # with only a single dimension. Gymnax requires calling the action_space
        # function while gymnasium does not.
        if isinstance(self.env, Environment):
            self.obs_shape = self.env.observation_space(self.env.default_params).shape
            self.action_space = self.env.action_space(self.env.default_params)
        elif isinstance(self.env, gymnasium.Env):
            self.obs_shape = self.env.observation_space.shape
            self.action_space = self.env.action_space
        else:
            raise ValueError("Unsupported environment type:", self.env)
        if isinstance(self.action_space, spaces.Discrete) or isinstance(self.action_space, gymnasium.spaces.Discrete):
            self.act_shape = int(self.action_space.n)
        elif isinstance(self.action_space, spaces.Box) or isinstance(self.action_space, gymnasium.spaces.Box):
            # Continuous case
            self.act_shape = int(np.prod(self.action_space.shape))
        else:
            raise ValueError("Unsupported action space type:", self.action_space)

        if self.logging:
            self.iteration_policy_entropy_ = []
            self.mean_discounted_returns_ = []
            self.sem_discounted_returns_ = []
            self.iteration_updated_tree_ = []
            self.mean_discretized_discounted_returns_ = []
            self.iterations_discretized_ = []
            self.ppo_losses_ = []

    def learn(self):
        # TODO: probably store num_actions and n_features_in to make all the lookups nicer

        env_init_rng, nn_init_rng, self.rng = jax.random.split(self.rng, num=3)

        if isinstance(self.env, Environment):
            obs, _ = self.env.reset(env_init_rng)
        else:
            obs, _ = self.env.reset()

        if self.max_leaf_nodes:
            max_nodes_leaves = self.max_leaf_nodes * 2 - 1
        else:
            max_nodes_leaves = 10000
        if self.max_depth:
            max_nodes_depth = 2 ** (self.max_depth + 1) - 1
        else:
            max_nodes_depth = 10000
        max_nodes = min(max_nodes_leaves, max_nodes_depth)
        self.tree_policy_ = DecisionTreePolicy(2 * self.env.num_actions, 2 * self.max_leaf_nodes - 1)
        self.rng, tree_init_rng = jax.random.split(self.rng)

        tree_params = self.tree_policy_.init(tree_init_rng, obs)

        self.tree_policy_.apply = jax.jit(self.tree_policy_.apply)

        if self.use_linear_value_function:
            self.value_network_ = ValueFunctionLinear()
        else:
            self.value_network_ = ValueNetwork()

        value_nn_state = TrainState.create(
            apply_fn=self.value_network_.apply,
            params=self.value_network_.init(nn_init_rng, obs),
            tx=optax.adam(learning_rate=2.5e-4),
        )
        self.value_network_.apply = jax.jit(self.value_network_.apply)

        self.rng, rng_rollout, rng_random_tree = jax.random.split(self.rng, num=3)
        random_observations = random_rollout_gymnax(
            self.env, self.env.default_params, rng_rollout, 100000
        )[0]

        self.rng, rollout_init_rng = jax.random.split(self.rng)
        self.n_steps = self.simulation_steps // self.num_envs
        self.rolloutmanager = ParallelEnvRollouts(
            self.env,
            num_envs=self.num_envs,
            n_steps=self.n_steps,
            gamma=self.gamma,
            gae_lambda=0.95,
        )
        curr_obs, curr_states = self.rolloutmanager.reset(rollout_init_rng)

        total_steps_collected = 0

        mean_entropy = jnp.sum(
            jax.scipy.special.entr(
                jnp.full(self.env.num_actions, fill_value=1 / self.env.num_actions)
            )
        )

        averaged_entropy = mean_entropy

        best_discretized_disc_return = -np.inf
        best_iteration = -1
        self.best_params_ = tree_params

        out_dir = "logs_dtpo_continuous"
        os.makedirs(out_dir, exist_ok=True)
        dtpo_continuous_json = os.path.join(out_dir, f"dtpo_continuous_results_{time.time()}.json")
        with open(dtpo_continuous_json, "w") as _:
            pass
        starting_time = time.time()

        # Main training loop
        for iteration in range(self.max_iterations):
            iteration_rng, self.rng = jax.random.split(self.rng)

            # Collect a batch of experience
            start_time = time.time()
            (
                observations_par,
                actions_par,
                rewards_par,
                done_par,
                curr_obs,
                curr_states,
            ) = self.rolloutmanager.get_batch(
                self.tree_policy_.apply,
                tree_params,
                curr_obs,
                curr_states,
                iteration_rng,
            )
            runtime = time.time() - start_time

            observations = observations_par.reshape(-1, observations_par.shape[2])
            # actions = actions_par.ravel()
            actions = actions_par.reshape(-1, self.act_shape)

            rewards = rewards_par.ravel()
            done = done_par.ravel()

            # Print the mean discounted reward
            pred_values = self.value_network_.apply(
                value_nn_state.params, observations
            ).ravel()
            returns = rlax.discounted_returns(
                r_t=rewards,
                discount_t=self.gamma * (1 - done),
                v_t=pred_values,
            )

            total_steps_collected += len(rewards)
            num_envs = self.rolloutmanager.num_envs
            n_steps = self.rolloutmanager.n_steps

            # Compute advantage for each time step with GAE(lambda)
            advantage_par = jax.vmap(
                rlax.truncated_generalized_advantage_estimation, in_axes=(1, 1, None, 1)
            )(
                rewards_par,
                self.gamma * (1 - done_par),
                0.95,
                jnp.pad(
                    pred_values.reshape(n_steps, num_envs), pad_width=((0, 1), (0, 0))
                ),
            )
            advantage = advantage_par.ravel(
                order="F"
            )  # We need F here to ensure the values are in the correct order

            # NOTE: this needs to be done before the advantage normalization
            target_values = advantage + pred_values

            # Normalize advantages
            if self.normalize_advantage:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            if self.verbose or self.logging:
                indices_done = jnp.where(done)[0]
                discounted_returns = returns[
                    jnp.concatenate((jnp.array([0]), indices_done[:-1] + 1))
                ]

                mean_discounted_return = jnp.mean(discounted_returns)
                sem_discounted_return = jnp.std(discounted_returns) / jnp.sqrt(
                    done.sum()
                )

            if self.logging:
                self.mean_discounted_returns_.append(mean_discounted_return.item())
                self.sem_discounted_returns_.append(sem_discounted_return.item())
                self.iteration_policy_entropy_.append(mean_entropy.item())

            if iteration % 100 == 0 and iteration != 0 and averaged_entropy < 0.01:
                # Done training once the policy is almost completely deterministic
                self.ppo_losses_.append(0.0)
                self.iteration_updated_tree_.append(False)
                break

            if self.verbose and iteration % 100 == 0:
                print(
                    f"Collected {len(observations)} steps of experience in {runtime} seconds"
                )

                return_summary = f"{mean_discounted_return} +- {sem_discounted_return}"
                print(
                    f"Policy update {iteration} mean return per iteration: {return_summary}"
                )

            # Update the value function network
            indices = jnp.arange(observations.shape[0])
            self.rng, rng_shuffle = jax.random.split(self.rng)

            batch_size = 64
            for epoch in range(4):
                indices = jax.random.permutation(rng_shuffle, indices)

                for i in range(0, len(indices), batch_size):
                    batch_indices = indices[i : i + batch_size]
                    value_nn_state = update_batch(
                        self.value_network_.apply,
                        value_nn_state,
                        batch_indices,
                        pred_values,
                        observations,
                        target_values,
                        self.ppo_epsilon,
                    )

            prev_logits = jax.vmap(self.tree_policy_.apply, in_axes=[None, 0])(
                tree_params, observations
            )
            prev_proba = jax.nn.softmax(prev_logits, axis=1)

            out_old = jax.vmap(self.tree_policy_.apply, in_axes=[None, 0])(
                tree_params, observations
            )
            mu_old, log_sigma_old = jnp.split(out_old, 2, axis=1)
            sigma_old = jnp.exp(log_sigma_old)
            prev_logp = jnp.sum(
                -0.5 * (
                    ((actions - mu_old) / sigma_old) ** 2 + 2 * log_sigma_old + jnp.log(2 * jnp.pi)
                ),
                axis=1,
            )

            mean_entropy = jnp.sum(jax.scipy.special.entr(prev_proba), axis=1).mean()

            if self.verbose and iteration % 100 == 0:
                runtime = time.time() - start_time

                print(f"After {runtime} seconds")
                print(f"Entropy: {averaged_entropy}")

            if iteration % 100 == 0:
                averaged_entropy = 0

            averaged_entropy += 0.01 * mean_entropy

            updated_tree = False

            self.iteration_updated_tree_.append(updated_tree)

            if iteration < self.warmup_iterations:
                # if set, in the first iterations only train the value function

                if self.logging:
                    self.ppo_losses_.append(0.0)

                continue

            self.rng, random_state_rng = jax.random.split(self.rng)
            random_state = jax.random.choice(random_state_rng, 100000).item()

            # Update the decision tree policy
            print("updating tree", iteration)
            ppo_loss_before = ppo_loss_samples_continuous(
                prev_logits,
                prev_logp,
                advantage,
                actions,
                self.ppo_epsilon,
            )

            best_loss = -jnp.inf
            best_tree_params = tree_params
            new_tree_params = tree_params
            best_update = -1

            if self.anneal_lr:
                lr = self.learning_rate * (1 - (iteration / self.max_iterations))
            else:
                lr = self.learning_rate

            frac = iteration / (self.max_iterations - 1)
            for update in tqdm(range(self.max_policy_updates)):
                new_tree_params, loss_after = learn_tree_structure_with_grad_batch(
                    observations,
                    advantage,
                    actions,
                    prev_logp,
                    self.tree_policy_.apply,
                    new_tree_params,
                    self.max_depth,
                    self.max_leaf_nodes,
                    random_state,
                    self.ppo_epsilon,
                    lr,
                    max_nodes,
                    frac,
                    self.grad_clip_norm,
                )

                if loss_after > best_loss:
                    best_loss = loss_after
                    best_tree_params = new_tree_params
                    best_update = update

            if self.verbose:
                print(f"best update: {best_update}, with loss: {best_loss}")

            # Only update the tree if it improves the PPO loss over the current tree
            if best_loss >= ppo_loss_before:
                tree_params = best_tree_params

            if self.logging:
                curr_logits = jax.vmap(self.tree_policy_.apply, in_axes=[None, 0])(
                    tree_params, observations
                )

                loss_value = ppo_loss_samples_continuous(
                    curr_logits,
                    prev_logp,
                    advantage,
                    actions,
                    self.ppo_epsilon,
                )
                self.ppo_losses_.append(loss_value.item())

            # Once every 10 iterations determinize and evaluate
            if iteration % 10 == 0:
                observations, rewards, done = deterministic_rollouts(
                    self.tree_policy_.apply,
                    tree_params,
                    self.simulation_steps,
                    iteration_rng,
                    self.env,
                )
                pred_returns = self.value_network_.apply(
                    value_nn_state.params, observations
                ).ravel()
                returns = rlax.discounted_returns(
                    r_t=rewards,
                    discount_t=self.gamma * (1 - done),
                    v_t=pred_returns,
                )
                indices_done = jnp.where(done)[0]
                if indices_done.shape[0] > 0:
                    first_done_index = int(indices_done[0])
                    undiscounted_return = float(jnp.sum(rewards[:first_done_index + 1]))
                else:
                    undiscounted_return = float(jnp.sum(rewards))
                discounted_returns = returns[
                    jnp.concatenate((jnp.array([0]), indices_done[:-1] + 1))
                ]
                mean_discounted_return = jnp.mean(discounted_returns).item()

                # Add JSON log for comparison with RPO
                now = time.time()
                relative_time = now - starting_time
                log_entry = {
                    "relative_time": float(relative_time),
                    "step": int(total_steps_collected),
                    "undiscounted_return": float(undiscounted_return),
                    "discounted_return": float(mean_discounted_return),
                }
                with open(dtpo_continuous_json, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

                if mean_discounted_return > best_discretized_disc_return:
                    best_discretized_disc_return = mean_discounted_return
                    best_iteration = iteration
                    self.best_params_ = tree_params

                if self.logging:
                    self.mean_discretized_discounted_returns_.append(
                        mean_discounted_return
                    )
                    self.iterations_discretized_.append(iteration)

                if self.verbose:
                    print(
                        f"Iteration {iteration} - discretized discounted return: {mean_discounted_return}"
                    )

            # Clear the JAX caches every few iterations to prevent memory leaks, e.g.
            # functions such as jax.lax.scan keep references that we need only once.
            # This does mean next iteration some extra compilation overhead occurs for
            # functions that we do reuse.
            if iteration % self.cache_clear_interval == 0 and iteration != 0:
                jax.clear_caches()

        if self.verbose:
            print("total steps collected:", total_steps_collected)
            print("best iteration:", best_iteration)

        self.discretized_tree_ = get_discretized_tree_continuous(
            self.best_params_,
            random_observations.shape[1],
            prune=True,
        )
