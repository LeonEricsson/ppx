import jax
import jax.numpy as jnp
from typing import Callable


def get_params_at_index(params, idx):
    return jax.tree_util.tree_map(lambda x: x[idx], params)


def evaluate_model(
    network_apply: Callable,
    params,
    env,
    n_trials: int,
    key: jax.Array,
    env_params=None,
):
    """
    Evaluate the model by running 'n_trials' amount of rollouts and return
    the mean and standard deviation of the collected resulting rewards.

    Args:
        action_fn: function(obs) -> action
        env: Gymnax-like environment
        n_trials: number of rollouts collected
        key: random seed
    """

    if env_params is None:
        env_params = env.default_params

    def action_fn(obs):
        return network_apply(params, obs)[0].sample_deterministic()

    @jax.jit
    def rollout(rng: jax.Array):

        def step(val):
            state, obs, rng, reward, _, goal_reached = val
            action = action_fn(obs)
            rng, _rng = jax.random.split(rng)
            new_obs, new_state, step_reward, done, info = env.step(
                _rng, state, action, env_params
            )

            return (new_state, new_obs, rng, reward + step_reward, done, jnp.logical_or(goal_reached, info['goal_reached']))

        def condition(val):
            _, _, _, _, done, _ = val
            return jnp.equal(done, False)

        rng, reset_rng = jax.random.split(rng)
        obs, state = env.reset(reset_rng, env_params)
        out = jax.lax.while_loop(
            condition, step, (state, obs, rng, 0, False, False))
        return out[3], out[5]

    seeds = jax.random.split(key, n_trials)

    episode_rewards, _ = jax.jit(jax.vmap(rollout))(seeds)

    won = jnp.sum(episode_rewards != -1) / n_trials

    return jnp.mean(episode_rewards), jnp.std(episode_rewards), won #jnp.sum(goal_reached) / n_trials
