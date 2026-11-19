# §3 MDP Formulation and Value Iteration

## 3.1 MDP formulation

Each item is modeled as a discounted MDP with $\gamma = 0.95$. Its size is set by the item's demand cap $D_{\max}$, the 99th percentile of training demand.

The state $(i, w)$ is the inventory $i \in \{0, \dots, 2D_{\max}\}$ at the start of the day and the weekday $w \in \{0, \dots, 6\}$, with 0 = Saturday. The action $a \in A(i) = \{0, \dots, \min(D_{\max},\ 2D_{\max} - i)\}$ is the number of units ordered that morning. The order arrives immediately, then demand $d \sim P(\cdot \mid w)$ occurs, and unmet demand is lost:

$$
\text{sales} = \min(i + a,\ d), \qquad \text{lost} = d - \text{sales}, \qquad i' = i + a - \text{sales}, \qquad w' = (w + 1) \bmod 7
$$

The reward is the day's profit:

$$
r(i, a, d) = p\,\text{sales} - c\,a - K\,\mathbb{1}[a > 0] - h\,i' - g\,\text{lost}
$$

M5 has no cost data, so costs are fixed fractions of the item's mean training price $p$: $c = 0.6p$, $h = 0.02p$, $K = 2p$, $g = 0.2p$.

| Item | M5 ID | $p$ ($) | $D_{\max}$ | States | Max actions |
| --- | --- | --- | --- | --- | --- |
| Low | FOODS_3_820 | 1.96 | 7 | 15 × 7 | 8 |
| Medium | FOODS_3_473 | 3.83 | 19 | 39 × 7 | 20 |
| High | FOODS_3_080 | 1.59 | 34 | 69 × 7 | 35 |

*Table 3.1: Item parameters and MDP sizes.*

We assume demand depends only on the weekday, not on inventory or past demand, and that orders have zero lead time. Because every reward term is a multiple of $p$, the optimal policy is independent of the item's price and depends only on its demand distribution and the cost fractions.

## 3.2 Demand models

We estimate $P(d \mid w)$ by maximum likelihood, which for a categorical distribution is the empirical frequency of each demand value:

$$
\hat P_{M0}(d) = \frac{1}{N} \sum_{t=1}^{N} \mathbb{1}[d_t = d], \qquad
\hat P_{M1}(d \mid w) = \frac{1}{N_w} \sum_{t:\, w_t = w} \mathbb{1}[d_t = d]
$$

M0 pools all $N$ training days, while M1 fits each weekday separately from its $N_w$ days and has seven times as many parameters. Weekday variation is substantial for the medium and high items, whose mean demand ranges from 6.4 to 10.3 and from 17.1 to 21.1 units across weekdays, peaking on weekends. For the low item it ranges only from 1.6 to 2.3. Both models are stored as $7 \times (D_{\max} + 1)$ tables, so value iteration runs unchanged on either one, or on a model refit from fewer days.

## 3.3 Value iteration

Starting from $V_0 = 0$, each sweep applies the Bellman optimality update to every state:

$$
V_{k+1}(i, w) = \max_{a \in A(i)} \sum_{d=0}^{D_{\max}} \hat P(d \mid w) \left[ r(i, a, d) + \gamma\, V_k(i', w') \right]
$$

After convergence, the policy takes the maximizing order in each state, breaking ties toward the smaller order. The reward and next inventory of every $(i, a, d)$ are precomputed with the same `step()` function used to evaluate every method, so value iteration’s rewards and inventory transitions match evaluation exactly; only the demand distribution is estimated.

Iteration stops when $\delta = \max_{i,w} |V_{k+1}(i, w) - V_k(i, w)| < 10^{-6}$. Since the Bellman operator is a $\gamma$-contraction in the max norm, this guarantees

$$
\lVert V_{k+1} - V^* \rVert_\infty \le \frac{\gamma}{1 - \gamma}\, \delta < 1.9 \times 10^{-5}.
$$

Convergence took between 174 and 316 sweeps, under a second per solve.

| Item | M0 sweeps | M1 sweeps |
| --- | --- | --- |
| Low | 174 | 225 |
| Medium | 308 | 314 |
| High | 313 | 316 |

*Table 3.2: Sweeps to convergence.*

## 3.4 Resulting policies

The resulting policies have a reorder-point structure: on weekday $w$, an order is placed only when inventory falls below a threshold $\theta_w$.

| Item | Model | Sat | Sun | Mon | Tue | Wed | Thu | Fri |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Low | M0 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| Low | M1 | 2 | 2 | 2 | 1 | 1 | 1 | 1 |
| Medium | M0 | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| Medium | M1 | 9 | 10 | 8 | 6 | 6 | 6 | 7 |
| High | M0 | 18 | 18 | 18 | 18 | 18 | 18 | 18 |
| High | M1 | 21 | 20 | 17 | 18 | 16 | 17 | 19 |

*Table 3.3: Reorder threshold $\theta_w$ (units) by weekday.*

M0 uses the same threshold every day. M1 raises it around the weekend peak and lowers it midweek; for the medium item, it reorders below 10 units on Sunday but below 6 on Tuesday through Thursday.

Order sizes are at or near the cap $a \le D_{\max}$: exactly $D_{\max}$ for the low and medium items and 29 to 34 units for the high item. Because the fixed cost $K = 2p$ is five times the per-unit margin $p - c = 0.4p$, and holding costs only $0.02p$ per unit per night, large infrequent orders are favored. With the cap raised to $2D_{\max}$, the M0 policy would order 13, 29 and 48 units from empty inventory rather than 7, 19 and 34. The reported policies are therefore optimal with respect to the order cap in our setup, not for unrestricted order sizes.

My role for this project was the implementation of MDP (step) and Value Iteration for M0 and M1. 
On my personal learning from this project, most of the core concepts were either covered in lecture or already familiar to me, so the project mostly reinforced existing knowledge rather than introducing something new. The main practical work was making sure value iteration planned with the same step function used to evaluate every method, and checking that its convergence error stayed within the theoretical bound.