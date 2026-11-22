# CSE 150A Final Project: Learning Inventory Restocking Policies

We model daily restocking of one grocery item as a Markov decision process and compare value iteration (with two demand models), tabular Q-learning, and a reorder-point baseline on three M5 items with low, medium, and high demand.

Run everything from the repo root.

## Setup

```bash
pip install numpy pandas matplotlib
```

The cleaned data for all three items is already committed in `data/processed/`, so you can run every method without downloading anything.

The raw M5 files are only needed to rebuild the processed data. To do that, download `calendar.csv`, `sales_train_evaluation.csv`, and `sell_prices.csv` from the [M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) competition into `data/raw/` (ignored by Git), then run `python data.py`.

## Quick start

```bash
python data.py                  # rebuild data/processed/ (needs raw M5 files)
python plots_data.py            # data figures -> figures/
python baseline.py              # reorder-point baseline, best (r, S) per item
python value_iteration.py       # VI with M0 and M1, test profit per item
python q_learning.py --tune     # tune alpha, train Q-learning, save results
python q_learning.py --plot     # convergence plots from saved results
python -m unittest test_q_learning
# [Tim: add the commands for the evaluation and experiment scripts]
```

## Files

| File | Owner | What it does |
| --- | --- | --- |
| `config.py` | Everyone | Shared settings: items, costs, caps, discount, Q-learning defaults |
| `data.py` | Cathlyn | Item selection, cleaning, train/test split, M0/M1 demand models |
| `plots_data.py` | Cathlyn | Data figures for the report |
| `baseline.py` | Cathlyn | Reorder-point baseline |
| `mdp.py` | Bryan | Shared costs and one-day transition used by every method |
| `value_iteration.py` | Bryan | Value iteration and exact policy evaluation |
| `q_learning.py` | Raul | Tabular Q-learning, tuning, convergence diagnostics |
| `test_q_learning.py` | Raul | Unit tests for Q-learning |
| [Tim's files] | Tim | [Evaluation and experiments] |

## config.py

All shared decisions live here, so everyone's code uses the same values:

- **Items:** store `CA_3`, department `FOODS_3`; low `FOODS_3_820`, medium `FOODS_3_473`, high `FOODS_3_080`
- **Data:** first 1,800 days train, the last 141 test; zero runs of 7+ days dropped; demand capped at the 99th percentile
- **Costs (fractions of price):** unit cost 0.6, holding 0.02, fixed order 2.0, lost sale 0.2
- **MDP:** `GAMMA = 0.95`, value iteration tolerance `VI_TOL = 1e-6`
- **Q-learning:** `Q_ALPHA = 0.1`, epsilon from 1.0 to 0.05, `Q_PASSES = 200`
- **Data-size experiment:** `DATA_SIZES = [60, 180, 365, 900, 1800]`

## data.py

```python
load_processed(level) -> dict
```
Loads one item's cleaned data from `data/processed/`. `level` is `"low"`, `"medium"`, or `"high"`. This is the function everyone else uses; it doesn't need the raw files.

Returns:
- `"train"`: `(demand, weekday)` arrays for the training days
- `"test"`: `(demand, weekday)` arrays for the 141 test days
- `"price"`: average selling price over training (float)
- `"d_max"`: demand cap (int)
- `"m0"`, `"m1"`: demand models, each shape `(7, d_max + 1)`

Weekdays are `0 = Saturday` through `6 = Friday`.

```python
fit_m0(demand, d_max) -> np.ndarray            # one distribution, repeated for all 7 weekdays
fit_m1(demand, weekday, d_max) -> np.ndarray   # one distribution per weekday
```
Maximum likelihood (empirical frequency) estimates, shape `(7, d_max + 1)`. Use these to refit a model on fewer days for the data-size experiment.

```python
build_processed()
```
Reads the raw M5 files and writes `data/processed/{low,medium,high}.npz`. Runs when you call `python data.py`.

## plots_data.py

Saves three figures to `figures/`: `demand_over_time.png`, `demand_hist_m0.png`, and `demand_by_weekday.png`.

## baseline.py

```python
order_up_to(demand, weekday, costs) -> (policy, r, S)
```
Tries every reorder point `r` and order-up-to level `S` with `r <= S <= i_max`, simulates each on the given demand, and returns the most profitable. The policy orders up to `S` only when inventory is below `r`.

```python
reorder_policy(r, S, i_max) -> policy          # policy[inventory, weekday], shape (i_max + 1, 7)
simulate(policy, demand, weekday, costs) -> float   # total profit, starting at 0 inventory
```

Example:
```python
from data import load_processed
from mdp import make_costs
from baseline import order_up_to

info = load_processed("medium")
costs = make_costs(info["price"], info["d_max"])
policy, r, S = order_up_to(*info["train"], costs)
```

## mdp.py

```python
make_costs(price, d_max, unit_frac=None, hold_frac=None, order_frac=None, lost_frac=None) -> dict
```
Costs and caps for one item. Fractions default to `config.py`; pass them to override (cost-sensitivity experiment).

```python
step(s, w, a, d, costs) -> (next_s, next_w, reward, lost)
```
One day: order `a` arrives, demand `d` happens. Orders above the caps are clipped.

## value_iteration.py

```python
value_iteration(probs, costs) -> (policy, V, n_iters)
```
`probs` is a demand model of shape `(7, d_max + 1)` (`info["m0"]`, `info["m1"]`, or refit with `fit_m0` / `fit_m1`).
`policy[inventory, weekday]` = order amount, shape `(2*d_max + 1, 7)`.

```python
policy_value(policy, probs, costs) -> V
```
Exact expected discounted profit of any policy under a demand model.

Example:
```python
from data import load_processed
from mdp import make_costs
from value_iteration import value_iteration

info = load_processed("medium")
costs = make_costs(info["price"], info["d_max"])
policy, V, n_iters = value_iteration(info["m1"], costs)
```

## q_learning.py

```python
q_learning(demand, weekday, costs, *, alpha=0.1, epsilon_start=1.0, epsilon_end=0.05,
           passes=200, gamma=0.95, seed=0, return_details=False) -> policy
```
Learns from the demand history in chronological passes, with epsilon decaying linearly by pass and inventory reset to 0 each pass. Returns `policy[inventory, weekday]`. With `return_details=True`, also returns the Q-table, visit counts, and per-pass diagnostics (TD error, largest Q change, policy change, coverage).

```python
tune_q_learning(demand, weekday, costs, *, validation_days=180,
                alphas=(0.05, 0.1, 0.2), seeds=(0, 1, 2)) -> list
```
Ranks learning rates by mean profit on the last `validation_days` of the **training** data, after training on the rest. Never touches test data. Refit on all training data with the chosen alpha afterward.

Command line:
```bash
python q_learning.py                       # default alpha, all items
python q_learning.py --tune                # tune alpha first
python q_learning.py --last-days 365       # train on the last 365 days only
python q_learning.py --levels medium --seed 1
python q_learning.py --plot                # plot saved convergence CSVs
```
Results (policies, Q-tables, tuning and convergence CSVs) are saved to `results/q_learning/` by default (`--output` to change).

## Evaluation and experiments