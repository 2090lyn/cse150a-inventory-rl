# Rahul — Q-learning report

## §3: Q-learning

We learned a tabular action-value function directly from the capped historical
training demands. State is (inventory, weekday); actions are integer order
quantities from zero through min(D_max, 2 D_max − inventory). Orders arrive before
demand, and each transition uses the project's sales, purchase, fixed ordering,
holding, and lost-sale reward. No M0 or M1 probability table is needed.

For each observed demand, the update is

Q(s,a) ← Q(s,a) + α [r + γ max_b Q(s′,b) − Q(s,a)].

The maximization includes only feasible actions. Feasible Q entries begin at
zero; infeasible entries are masked with negative infinity. Training uses
α=0.1, γ=0.95, and 200 chronological passes through days 1–1800. Epsilon decreases
linearly from 1.0 on pass 1 to 0.05 on pass 200. Exploration samples feasible
orders uniformly; exploitation randomly breaks Q-value ties. Inventory starts
at zero each pass. The final deterministic policy takes the maximizing feasible
order at every state, breaking ties toward the smallest order. The reported
runs use random seed 0. Every policy has shape (2 D_max + 1, 7).

The data boundary is treated as a truncation of a continuing discounted problem:
the final observation still bootstraps to the next weekday's Q-value. No
transition from the last demand to the first demand is simulated. If cleanup
removes calendar days, the bootstrap weekday is the next retained observation's
weekday, and no artificial demands are inserted for excluded days. For these
three processed items all 1800 training observations are retained.

As a separate sensitivity experiment, we compared α in {0.05, 0.1, 0.2} using
three seeds {0,1,2}. The first 1620 training observations supplied learning data,
and the final 180 supplied validation demand. Each validation simulation started
with zero inventory and used a deterministic greedy policy. Mean validation
profit selected α; the selected configuration was then refit on all 1800
observations. The official test set was never loaded into training or tuning.
The shared preprocessing cap and mean price are derived from the full training
period, so this holdout study is conditional on those fixed preprocessing
parameters; it is not a fully nested preprocessing validation experiment.

| Item   | Selected α | Mean validation profit | Seed standard deviation |
| ------ | ---------: | ---------------------: | ----------------------: |
| Low    |       0.05 |                  39.60 |                    0.00 |
| Medium |       0.20 |                1204.77 |                   32.60 |
| High   |       0.20 |                1013.29 |                   17.12 |

These are holdout profits in the project's relative currency units, not test
results. Keep the required α=0.1 results separate from the tuned variant.

## §4: Convergence diagnostics

We recorded the mean absolute sampled TD error, largest absolute Q-table change
in each pass, fraction of greedy policy entries changed since the preceding
pass, and fraction of feasible state-action entries visited at least once.
The figure `results/q_learning/default/convergence.png` shows the required
α=0.1 runs; CSV histories accompany it.

| Item   | Policy shape | Final mean absolute TD error | Final maximum Q change | Final policy change | Feasible entries visited |
| ------ | ------------ | ---------------------------: | ---------------------: | ------------------: | -----------------------: |
| Low    | 15 × 7       |                        0.398 |                  0.283 |               0.00% |                  100.00% |
| Medium | 39 × 7       |                        4.416 |                  5.977 |               0.37% |                   99.95% |
| High   | 69 × 7       |                        4.482 |                 11.712 |               0.83% |                   96.79% |

The final-pass greedy policies changed little, but Q-values continued to move.
These observations support partial policy stabilization rather than proof of
convergence. Fixed α, finite replay, remaining exploration, and incomplete
coverage constrain such a claim. A sampled TD error can remain nonzero even
for a useful value estimate because individual demand observations vary.
The larger inventory/action space for the high-demand item also makes coverage
harder within a fixed update budget. Absolute TD errors are in item-specific
reward units and should not be compared as normalized errors across items.
Tim's held-out evaluation and seed sensitivity are still needed to assess
policy quality relative to value iteration and the baseline.

## Contribution and reflection draft — review before submitting

Raph's assigned contribution is the historical-demand Q-learning implementation,
training-only hyperparameter study, reproducible policy exports, convergence
diagnostics, and the associated report sections. The implementation emphasizes
feasible actions and separates default settings from tuned settings. A key
lesson from these runs is that a stable greedy policy can coexist with changing
Q-values, and that high-dimensional tables need explicit coverage diagnostics.
Edit this paragraph to reflect your own review, understanding, and actual work;
it is a draft, not a claim of unaided authorship.

## Reference

Watkins, C. J. C. H., & Dayan, P. (1992). Technical Note: Q-Learning.
_Machine Learning, 8_, 279–292.
https://doi.org/10.1023/A:1022676722315
Author-hosted paper: https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf

The paper establishes convergence under its sampling and learning-rate
conditions; our fixed-rate, finite-data experiment does not establish those
conditions.
