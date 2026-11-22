import numpy as np

import config
from data import load_processed


from mdp import make_costs, step


def reorder_policy(r, S, i_max):
    """Order up to S only when inventory is below r. Shape (i_max + 1, 7)."""
    policy = np.zeros((i_max + 1, 7), dtype=int)
    for s in range(i_max + 1):
        if s < r:
            policy[s, :] = S - s
    return policy


def simulate(policy, demand, weekday, costs):
    """Run a policy over a demand sequence starting at 0 inventory. Returns total profit."""
    s, total = 0, 0.0
    for d, w in zip(demand, weekday):
        a = policy[s, w]
        s, _, reward, _ = step(s, w, a, d, costs)
        total += reward
    return total


def order_up_to(demand, weekday, costs):
    """Try every (r, S) with r <= S, keep the most profitable on training data.
    Returns (policy, best_r, best_S)."""
    best, best_profit = (0, 0), -np.inf
    for r in range(costs["i_max"] + 1):
        for S in range(r, costs["i_max"] + 1):
            profit = simulate(reorder_policy(r, S, costs["i_max"]), demand, weekday, costs)
            if profit > best_profit:
                best, best_profit = (r, S), profit
    r, S = best
    return reorder_policy(r, S, costs["i_max"]), r, S


if __name__ == "__main__":
    for level in ["low", "medium", "high"]:
        info = load_processed(level)
        costs = make_costs(info["price"], info["d_max"])
        train_d, train_w = info["train"]
        policy, r, S = order_up_to(train_d, train_w, costs)
        print(f"{level}: best r = {r}, best S = {S}, d_max = {info['d_max']}, "
                f"train profit = {simulate(policy, train_d, train_w, costs):.2f}")