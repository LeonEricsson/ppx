import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import distrax
from flax.linen.initializers import constant, orthogonal
import functools
from typing import Sequence, Tuple, Dict


class TupleCategorical:
    distributions: Tuple[distrax.Distribution, distrax.Distribution]

    def __init__(self, logits_first, logits_second):

        self.distributions = (
            distrax.Categorical(logits=logits_first),
            distrax.Categorical(logits=logits_second)
        )

    # Sample an action
    def sample(self, seed):

        rngs = jax.random.split(seed)

        return (
            self.distributions[0].sample(seed=rngs[0]),
            self.distributions[1].sample(seed=rngs[1])
        )

    # Return the best action
    def sample_deterministic(self):
        return (
            jnp.argmax(self.distributions[0].probs),
            jnp.argmax(self.distributions[1].probs),
        )

    def log_prob(self, value):
        return self.distributions[0].log_prob(value[0]) \
            + self.distributions[1].log_prob(value[1])

    def entropy(self):
        return jnp.sum(jnp.array([dist.entropy() for dist in self.distributions]))


class ActorCriticTuple(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @ nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)

        logits_first = nn.Dense(
            self.action_dim[0], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        logits_second = nn.Dense(
            self.action_dim[1], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = TupleCategorical(logits_first, logits_second)

        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        features = carry[0].shape[-1]
        new_rnn_state, y = nn.GRUCell(features)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell(
            hidden_size,
            parent=None,
            kernel_init=orthogonal(1),
            recurrent_kernel_init=orthogonal(1)).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(64, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(64, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)
