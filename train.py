from src.ppo import make_train_envparams, Hparams, ActorCriticTuple
import jax
import pickle
from src.utils import evaluate_model

if __name__ == "__main__":
    config = {
        "NUM_ENVS": 16,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 1e5,
        "UPDATE_EPOCHS": 5,
        "NUM_MINIBATCHES": 32,
        "ENV_NAME": None, # Environment
        "NORMALIZE_OBS": False,  # If this is used, we must save the resulting normalization
        "ANNEAL_LR": True,
    }

    key = jax.random.PRNGKey(42)
    # num_seeds = 32

    hparams = Hparams(
        lr=1e-4,
        gamma=0.99,
        gae_lambda=0.8,
        clip_eps=0.3,
        ent_coef=0.00023,
        vf_coef=0.476,
        max_grad_norm=0.7,
    )
    print("Using hparams:\n", hparams)

    env = None
    env_params = env.default_params

    network = ActorCriticTuple(
        (
            env.action_space(env_params).spaces[0].n,
            env.action_space(env_params).spaces[1].n,
        ),
        activation="relu",
    )

    train = jax.jit(make_train_envparams(config))

    outs = jax.block_until_ready(train(hparams, key, env_params))

    network_params = outs["runner_state"][0].params

    eval_mean, eval_std = evaluate_model(
        network.apply, network_params, env, 1000, key, env_params
    )

    print(f"{eval_mean} ± {eval_std}")

    with open("trained_models/params_pattern_6.pickle", "wb") as file:
        pickle.dump(network_params, file)
