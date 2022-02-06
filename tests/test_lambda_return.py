import jax
import rlax
import matplotlib.pyplot as plt
import jax.numpy as jnp


# 1D case
rewards = jnp.zeros(50)
rewards = rewards.at[40].set(1)
discounts = jnp.full_like(rewards, 0.86)
estimated_returns = jnp.zeros(50)
lambdas = jnp.full_like(rewards, 1.0)
target_returns = rlax.lambda_returns(rewards, discounts, estimated_returns, lambdas)

plt.plot(rewards)
plt.plot(target_returns)
plt.show()


# 2D case
rewards = jnp.zeros((2, 50))
rewards = rewards.at[0, 40].set(1)
rewards = rewards.at[1, 10].set(1)
discounts = jnp.full_like(rewards, 0.86)
estimated_returns = jnp.zeros((2, 50))
estimated_returns = estimated_returns.at[0, 25].set(1)
lambdas = jnp.full_like(rewards, 1.0)

batched = jax.vmap(
    lambda rewards, discounts, estimated_returns, lambdas:
        rlax.lambda_returns(rewards, discounts, estimated_returns, lambdas, stop_target_gradients=True)
)

target_returns = batched(rewards, discounts, estimated_returns, lambdas)

plt.plot(rewards.T)
plt.plot(target_returns.T)
plt.show()


# target_returns = batched(rewards, discounts, best_recomputed_returns, lambdas)
