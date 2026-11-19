Run everything from the repo root.

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