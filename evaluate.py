import numpy as np
from mdp import step

def evaluate(policy, demand, weekday, costs, start_inventory=0):
    policy = np.asarray(policy)
    s = start_inventory

    profit          = 0
    lost            = 0
    stockouts       = 0
    orders          = 0
    inv_sum         = 0
    total_demand    = 0
    for d, w in zip(demand, weekday):
        a = int(policy[s, int(w)])
        s, _, reward, l = step(s, int(w), a, int(d), costs)
        profit += reward
        lost += l
        if l > 0:
            stockouts += 1
        if a > 0:
            orders += 1
        inv_sum += s
        total_demand += min(int(d), costs["d_max"])
    n = len(demand)
    return {
        "profit": float(profit),
        "profit_per_day": float(profit) / n,
        "lost_units": int(lost),
        "stockout_days": int(stockouts),
        "fill_rate": 1 - lost / total_demand if total_demand else 1.0,
        "orders": int(orders),
        "avg_inventory": inv_sum / n,
        "days": n,
    }
