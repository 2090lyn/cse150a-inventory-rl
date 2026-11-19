import numpy as np
import config
from mdp import make_costs, max_order, step

# weekday after w
NEXT_W = (np.arange(7) + 1) % 7          

def build_tables(costs):
    """run step() on every (s, a, d) once and save the results so we don't
    redo it every sweep. uses step() itself so the model can't disagree with
    the simulator. weekday doesn't change what happens, just the odds, so no w axis.

    Returns 
        reward[s, a, d]: that day's reward (0 if the order isn't allowed)
        next_s[s, a, d]: tomorrow's inventory (0 if the order isn't allowed)
        legal[s, a]: is this order allowed
    """
    n_s, n_a, n_d = costs["i_max"] + 1, costs["a_max"] + 1, costs["d_max"] + 1
    reward = np.zeros((n_s, n_a, n_d))
    next_s = np.zeros((n_s, n_a, n_d), dtype=int)
    legal = np.zeros((n_s, n_a), dtype=bool)
    for s in range(n_s):
        for a in range(max_order(s, costs) + 1):
            legal[s, a] = True
            for d in range(n_d):
                next_s[s, a, d], _, reward[s, a, d], _ = step(s, 0, a, d, costs)
    return reward, next_s, legal

def _q_values(V, probs, reward, next_s, legal, gamma):
    """Q[s, a, w] = sum_d probs[w, d] * (reward[s, a, d] + gamma * V[next_s[s, a, d], w + 1])."""
    V_tomorrow = V[:, NEXT_W]                                   
    future = V_tomorrow[next_s]                                 
    target = reward[..., None] + gamma * future                 
    Q = np.einsum("sadw,wd->saw", target, probs)             
    Q[~legal] = -np.inf                                        
    return Q


def value_iteration(probs, costs, gamma=config.GAMMA, tol=config.VI_TOL, max_iters=100_000):
    """Apply the Bellman optimality update until max |V_new - V| < tol.

    Returns (policy, V, n_iters)
        policy[s, w] : order amount, shape (i_max + 1, 7)
        V[s, w]: expected discounted profit from (s, w)
    """
    probs = np.asarray(probs, dtype=float)
    assert probs.shape == (7, costs["d_max"] + 1), "probs must be (7, d_max + 1)"
    assert np.allclose(probs.sum(axis=1), 1), "each weekday's probs must sum to 1"

    reward, next_s, legal = build_tables(costs)
    V = np.zeros((costs["i_max"] + 1, 7))
    for it in range(1, max_iters + 1):
        V_new = _q_values(V, probs, reward, next_s, legal, gamma).max(axis=1)
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < tol:
            break

    policy = _q_values(V, probs, reward, next_s, legal, gamma).argmax(axis=1)  # ties -> smaller order
    return policy.astype(int), V, it


def policy_value(policy, probs, costs, gamma=config.GAMMA):
    """Exact value of any policy under a demand model (solves V = r + gamma P V). Used to check 
    value iteration and to compare policies (e.g. the baseline) under the same model."""
    reward, next_s, _ = build_tables(costs)
    n_s = costs["i_max"] + 1
    n = n_s * 7
    idx = lambda s, w: s * 7 + w
    P = np.zeros((n, n))
    r = np.zeros(n)
    for s in range(n_s):
        for w in range(7):
            a = min(int(policy[s, w]), costs["a_max"], costs["i_max"] - s)
            r[idx(s, w)] = probs[w] @ reward[s, a]
            for d, pd in enumerate(probs[w]):
                P[idx(s, w), idx(next_s[s, a, d], NEXT_W[w])] += pd
    V = np.linalg.solve(np.eye(n) - gamma * P, r)
    return V.reshape(n_s, 7)


def solve(info, model="m1"):
    costs = make_costs(info["price"], info["d_max"])
    return value_iteration(info[model], costs)


if __name__ == "__main__":
    from data import load_processed

    WEEKDAYS = ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"]

    def run(policy, demand, weekday, costs):
        s, profit, lost, stockouts = 0, 0.0, 0, 0
        for d, w in zip(demand, weekday):
            s, _, r, l = step(s, w, policy[s, w], d, costs)
            profit, lost, stockouts = profit + r, lost + l, stockouts + (l > 0)
        return profit, lost, stockouts

    for level in ["low", "medium", "high"]:
        info = load_processed(level)
        costs = make_costs(info["price"], info["d_max"])
        test_d, test_w = info["test"]
        print(f"\n=== {level}: d_max = {info['d_max']}, p = {info['price']:.2f} ===")
        for model in ["m0", "m1"]:
            policy, V, n_iters = value_iteration(info[model], costs)
            profit, lost, stockouts = run(policy, test_d, test_w, costs)
            print(f"{model.upper()}: {n_iters} iters | test profit {profit:8.2f} | "
                  f"lost units {lost:4d} | stockout days {stockouts:3d}/{len(test_d)}")
            print("order at s=0 by weekday:", dict(zip(WEEKDAYS, policy[0].tolist())))
