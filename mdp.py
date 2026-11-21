import config

def make_costs(price, d_max, unit_frac=None, hold_frac=None, order_frac=None, lost_frac=None):
    """Dollar costs + caps for one item."""
    unit_frac = config.UNIT_COST_FRAC if unit_frac is None else unit_frac
    hold_frac = config.HOLD_FRAC if hold_frac is None else hold_frac
    order_frac = config.ORDER_COST_FRAC if order_frac is None else order_frac
    lost_frac = config.LOST_FRAC if lost_frac is None else lost_frac
    return {
        "p": price,
        "c": unit_frac * price,
        "h": hold_frac * price,
        "K": order_frac * price,
        "g": lost_frac * price,
        "d_max": d_max,
        "i_max": 2 * d_max, # largest possible inventory     
        "a_max": d_max, # largest possible order          
    }

def max_order(s, costs):
    return min(costs["a_max"], costs["i_max"] - s)

def step(s, w, a, d, costs):
    """One day: order arrives, demand happens. Returns (next_s, next_w, reward, lost). 
    Orders above caps are clipped."""
    a = int(min(max(a, 0), max_order(s, costs)))
    d = int(min(d, costs["d_max"]))
    sales = min(s + a, d)
    lost = d - sales
    next_s = s + a - sales
    reward = (costs["p"] * sales - costs["c"] * a
              - costs["K"] * (a > 0) - costs["h"] * next_s - costs["g"] * lost)
    return next_s, (w + 1) % 7, reward, lost