# Inventory reinforcement learning

CA_3 / FOODS_3 inventory control. Shared parameters and locked item IDs are in
`config.py`. Cleaned item data are loaded with `data.load_processed(level)`.

## Q-learning (rAHUL)

Install dependencies in a virtual environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install numpy pandas matplotlib
.venv/bin/python q_learning.py --output results/q_learning/default
.venv/bin/python -m unittest -v test_q_learning
```

```python
from data import load_processed
from q_learning import make_costs, q_learning
info = load_processed('low')
costs = make_costs(info['price'], info['d_max'])
policy = q_learning(*info['train'], costs)
# policy[inventory, weekday] -> feasible integer order
policy, details = q_learning(*info['train'], costs, return_details=True)
# details: q, visits, history (one diagnostics dictionary per pass)
```

Weekdays use the existing M5 convention, 0=Saturday through 6=Friday.
Defaults are alpha=0.1, gamma=0.95, 200 passes, linear epsilon 1.0 to 0.05,
seed=0. Each pass starts with zero inventory. Greedy training ties are random;
exported policy ties choose the smallest order. Infeasible actions are masked.
No demand distribution is fitted for Q-learning; it replays observed demand.
A local transition matches `baseline.step` while `mdp.py` is absent; when
available, `mdp.step(s, w, a, d, costs)` is imported automatically and must return
`(next_inventory, next_weekday, reward, lost)`.

Tuning compares alpha 0.05/0.1/0.2 over seeds 0/1/2 using a chronological holdout
of min(180, N//5) training observations, and then retrains with seed 0 (or --seed).
Keep default and tuned methods labeled separately. NPZ files contain policy,
Q, visitation counts, and run parameters. CSV files contain convergence history
and, for tuning runs, validation mean/std profit. Run these commands from the
repository root. Diagnostics refer to exploratory training returns, not test
performance. Tim's evaluator can load `np.load(path)['policy']` directly.

`--last-days` slices retained observations because processed files contain no
original day indices. If any days were excluded, this differs from a window of
calendar days; coordinate that distinction for the less-data experiment.
The existing full-training cap and price remain fixed for these slices.
See `report_q_learning.md` for report text, results, and limitations.

Regenerate the convergence figure from the saved histories:

```sh
python q_learning.py --plot --output results/q_learning/default
```
