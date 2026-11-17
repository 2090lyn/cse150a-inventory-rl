"""Tabular Q-learning on chronological, cleaned historical demand."""
import argparse
import csv
from pathlib import Path

import numpy as np

import config


def make_costs(price, d_max):
    """Same cost dictionary consumed by baseline.step and the shared MDP."""
    return dict(p=price, c=config.UNIT_COST_FRAC * price,
                h=config.HOLD_FRAC * price, K=config.ORDER_COST_FRAC * price,
                g=config.LOST_FRAC * price, i_max=2 * d_max, a_max=d_max)


def _step(s, w, a, d, costs):
    # Local until the shared mdp.py lands; match baseline.py's equations.
    sales = min(s + a, d)
    lost = d - sales
    remaining = s + a - sales
    reward = (costs['p'] * sales - costs['c'] * a - costs['K'] * (a > 0)
              - costs['h'] * remaining - costs['g'] * lost)
    return remaining, (w + 1) % 7, reward, lost


def _transition():
    try:
        from mdp import step
    except ModuleNotFoundError as exc:
        if exc.name != 'mdp':
            raise
        return _step
    return step


def _inputs(demand, weekday, costs):
    d, w = np.asarray(demand), np.asarray(weekday)
    if d.ndim != 1 or d.size == 0 or w.shape != d.shape:
        raise ValueError('demand and weekday must be nonempty equal-length vectors')
    for values, upper, name in [(d, costs['a_max'], 'demand'), (w, 6, 'weekday')]:
        if not np.all(np.isfinite(values)) or np.any(values != np.floor(values)) or np.any(values < 0) or np.any(values > upper):
            raise ValueError(f'{name} must contain integers in [0, {upper}]')
    return d.astype(int), w.astype(int)


def q_learning(demand, weekday, costs, *, alpha=config.Q_ALPHA,
               epsilon_start=config.Q_EPS_START, epsilon_end=config.Q_EPS_END,
               passes=config.Q_PASSES, gamma=config.GAMMA, seed=0,
               return_details=False):
    """Return policy[inventory, weekday]; optionally also Q, visits, and history.

    Epsilon decays linearly by pass. Inventory resets to zero each pass.
    Excluded calendar days are not simulated: the next retained observation's
    weekday is the successor weekday. The last observation bootstraps to the
    next calendar weekday (continuing discounted task, not a terminal reward).
    """
    for key in ('i_max', 'a_max'):
        if not isinstance(costs[key], (int, np.integer)) or costs[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    if not (0 < alpha <= 1 and 0 <= gamma < 1 and
            0 <= epsilon_end <= epsilon_start <= 1):
        raise ValueError('invalid learning rate, discount, or epsilon range')
    if not isinstance(passes, (int, np.integer)) or passes < 1:
        raise ValueError('passes must be a positive integer')
    d, w = _inputs(demand, weekday, costs)
    rng = np.random.default_rng(seed)
    i_max, a_max = costs['i_max'], costs['a_max']
    limits = np.minimum(a_max, i_max - np.arange(i_max + 1)) + 1
    feasible = np.arange(a_max + 1)[None, None, :] < limits[:, None, None]
    q = np.where(np.broadcast_to(feasible, (i_max + 1, 7, a_max + 1)), 0., -np.inf)
    visits = np.zeros(q.shape, dtype=np.int64)
    history = []
    step = _transition()
    for epoch in range(passes):
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * epoch / max(passes - 1, 1)
        previous = q.copy()
        old_policy = q.argmax(axis=2)
        s, total, td_sum = 0, 0., 0.
        for t, (dt, wt) in enumerate(zip(d, w)):
            n = limits[s]
            row = q[s, wt, :n]
            if rng.random() < epsilon:
                a = int(rng.integers(n))
            else:
                # Random ties prevent zero-initialization from favoring no order.
                a = int(rng.choice(np.flatnonzero(row == row.max())))
            ns, nw, reward, _ = step(s, int(wt), a, int(dt), costs)
            if t + 1 < len(w):
                nw = int(w[t + 1])
            delta = reward + gamma * q[ns, nw, :limits[ns]].max() - q[s, wt, a]
            q[s, wt, a] += alpha * delta
            visits[s, wt, a] += 1
            total += reward
            td_sum += abs(delta)
            s = ns
        policy = q.argmax(axis=2)
        mask = np.broadcast_to(feasible, q.shape)
        history.append(dict(pass_number=epoch + 1, epsilon=epsilon,
                            training_profit=total, mean_abs_td=td_sum / len(d),
                            max_q_change=float(np.max(np.abs(q[mask] - previous[mask]))),
                            policy_change_fraction=float(np.mean(policy != old_policy)),
                            visited_fraction=float(np.count_nonzero(visits) / np.count_nonzero(mask))))
    if return_details:
        return policy, dict(q=q, visits=visits, history=history)
    return policy


def _profit(policy, demand, weekday, costs):
    s, total = 0, 0.
    step = _transition()
    for d, w in zip(demand, weekday):
        s, _, reward, _ = step(s, int(w), int(policy[s, w]), int(d), costs)
        total += reward
    return total


def tune_q_learning(demand, weekday, costs, *, validation_days=180,
                    alphas=(0.05, 0.1, 0.2), seeds=(0, 1, 2), passes=config.Q_PASSES):
    """Rank alpha values by mean profit on a training-only chronological holdout.

    Call q_learning on ALL training data with the chosen alpha afterward.
    No test demand is accepted or consulted by this helper.
    """
    d, w = _inputs(demand, weekday, costs)
    if not 0 < validation_days < len(d) or not seeds or not alphas:
        raise ValueError('need a nonempty training prefix, holdout, candidates, and seeds')
    results = []
    for alpha in alphas:
        scores = []
        for seed in seeds:
            policy = q_learning(d[:-validation_days], w[:-validation_days], costs,
                                alpha=alpha, seed=seed, passes=passes)
            scores.append(_profit(policy, d[-validation_days:], w[-validation_days:], costs))
        results.append(dict(alpha=alpha, mean_validation_profit=float(np.mean(scores)),
                            std_validation_profit=float(np.std(scores))))
    return sorted(results, key=lambda row: row['mean_validation_profit'], reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--levels', nargs='+', choices=list(config.ITEMS), default=list(config.ITEMS))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--passes', type=int, default=config.Q_PASSES)
    parser.add_argument('--last-days', type=int, choices=config.DATA_SIZES)
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--plot', action='store_true', help='plot existing convergence CSVs and exit')
    parser.add_argument('--output', default='results/q_learning')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.plot:
        plot_convergence(output)
        return
    from data import load_processed
    for level in args.levels:
        info = load_processed(level)
        d, w = info['train']
        if args.last_days:
            d, w = d[-args.last_days:], w[-args.last_days:]
        costs = make_costs(info['price'], info['d_max'])
        stem = f'{level}_{args.last_days or "full"}_seed{args.seed}'
        alpha = config.Q_ALPHA
        if args.tune:
            ranking = tune_q_learning(d, w, costs, validation_days=min(180, len(d) // 5), passes=args.passes)
            alpha = ranking[0]['alpha']
            _write_csv(output / f'{stem}_tuning.csv', ranking)
        policy, details = q_learning(d, w, costs, alpha=alpha, passes=args.passes,
                                     seed=args.seed, return_details=True)
        np.savez_compressed(output / f'{stem}.npz', policy=policy, q=details['q'],
                            visits=details['visits'], alpha=alpha, seed=args.seed,
                            passes=args.passes, gamma=config.GAMMA, training_days=len(d))
        _write_csv(output / f'{stem}_convergence.csv', details['history'])
        print(f'{level}: policy {policy.shape}, {len(d)} training observations, alpha={alpha}, '
              f'visited={details["history"][-1]["visited_fraction"]:.1%}')


def plot_convergence(output):
    """Plot diagnostics already saved by training; does not rerun training."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files = sorted(output.glob('*_convergence.csv'))
    if not files:
        raise ValueError(f'No convergence CSVs in {output}')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    metrics = [('mean_abs_td', 'Mean absolute TD error'),
               ('max_q_change', 'Maximum Q change per pass'),
               ('policy_change_fraction', 'Fraction of policy entries changed'),
               ('visited_fraction', 'Feasible state-action coverage')]
    for path in files:
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        for ax, (metric, label) in zip(axes.flat, metrics):
            ax.plot([int(r['pass_number']) for r in rows],
                    [float(r[metric]) for r in rows], label=path.stem.split('_')[0])
            ax.set(xlabel='Training pass', ylabel=label)
            ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.savefig(output / 'convergence.png', dpi=160)
    plt.close(fig)


def _write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
