import jax
import jax.numpy as jnp


def transform_to_range(val, minval, maxval):
    return minval + (maxval - minval) * val


def make_es_step(f, transform=lambda w: w, sigma_decay=0.99, sigma_limit=0.01, alpha=0.05, npop=64):

    def step(carry, _):

        w, key, sigma = carry

        key, subkey = jax.random.split(key)

        N = sigma * jax.random.normal(subkey, (npop, *w.shape))

        # Mutate our weights into a new population
        w_mutated = jnp.clip(w + N, a_min=0.0, a_max=1.0)

        # Transform the weights
        w_transformed = jax.vmap(transform)(w_mutated)

        key, subkey = jax.random.split(key)

        # Evaluate our new weights
        results = f(w_transformed, subkey)

        # Normalized results
        A = (results - jnp.mean(results)) / jnp.max(jnp.array([jnp.std(results), 1e-1]))

        # Step our weight in the gradient direction
        w = jnp.clip(w + (alpha / (npop * sigma)) * jnp.dot(N.T, A), a_min=0.0, a_max=1.0)

        # Update our sigma (variance of mutations)
        sigma = jnp.max(jnp.array([sigma * sigma_decay, sigma_limit]))

        return (
            (w, key, sigma),
            (jnp.mean(results), jnp.max(results), w_mutated[jnp.argmax(results)])
        )

    return step


if __name__ == "__main__":

    # EXAMPLE OF RUNNING THE ES ALGORITHM

    def transform(w):
        """ Transform for a single weight before input into evaluation function """
        return w

    def f(w_transformed, key):
        """
        Batch of transformed w's -> vector of results of those w's

        This function will be maximized in the ES algorithm
        """
        return -jnp.linalg.norm(
            w_transformed - jnp.array([0.5, 0.5, 0.5, 0.5]),
            ord=2,
            axis=1
        )

    iterations = 50
    npop = 64
    start_sigma = 0.04
    sigma_decay = 0.99
    sigma_limit = 0.01
    alpha = 0.05

    key = jax.random.PRNGKey(42)

    # Initialize randomly
    key, subkey = jax.random.split(key)

    # Start guess ( may be a random guess as well)
    w = jnp.array([0.1, 0.8, 0.6, 0.0])

    # Create our single update in the es algorithm
    # es_step = jax.jit(make_es_step(f, transform, sigma_decay, sigma_limit, alpha, npop))
    es_step = jax.jit(make_es_step(f))

    # Run the algorithm
    # Also supports scanning over steps but that prevents printing
    sigma = start_sigma
    for i in range(iterations):
        (w, key, sigma), (mean, max, best_w) = es_step((w, key, sigma), None)
        print("Iteration:", i, "mean:", mean, "w:", w)

    print("Final w:", transform(w), "Eval:", f(jnp.array([w]), None)[0])
