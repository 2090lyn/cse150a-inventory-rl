import os

import numpy as np
import pandas as pd

import config

day_cols = [f"d_{i}" for i in range(1, 1942)]


def long_zero_mask(demand, min_len):
    """True for days inside a zero run of length >= min_len."""
    mask = np.zeros(len(demand), dtype=bool)
    start = None
    for i, x in enumerate(np.append(demand, 1)):          # extra 1 closes a final run
        if x == 0 and start is None:
            start = i
        elif x != 0 and start is not None:
            if i - start >= min_len:
                mask[start:i] = True
            start = None
    return mask


def fit_m0(demand, d_max):
    """One distribution for all days, repeated for each weekday. Shape (7, d_max + 1)."""
    counts = np.bincount(demand, minlength=d_max + 1)
    probs = counts / counts.sum()
    return np.tile(probs, (7, 1))


def fit_m1(demand, weekday, d_max):
    """One distribution per weekday. Shape (7, d_max + 1)."""
    m1 = np.zeros((7, d_max + 1))
    for w in range(7):
        counts = np.bincount(demand[weekday == w], minlength=d_max + 1)
        m1[w] = counts / counts.sum()
    return m1


def build_processed():
    """Read the raw M5 files and save cleaned data for each item. Only Cathlyn needs to run this."""
    sales = pd.read_csv(f"{config.RAW_DIR}/sales_train_evaluation.csv")
    ca3_foods = sales[(sales["store_id"] == config.STORE) & (sales["dept_id"] == config.DEPT)]
    calendar = pd.read_csv(f"{config.RAW_DIR}/calendar.csv")
    prices = pd.read_csv(f"{config.RAW_DIR}/sell_prices.csv")
    weekday_all = calendar["wday"].to_numpy()[:1941] - 1     # 0 = Sat ... 6 = Fri

    def load_item(item_id):
        row = ca3_foods[ca3_foods["item_id"] == item_id].iloc[0]
        demand = row[day_cols].to_numpy(dtype=int)

        # split
        n = config.TRAIN_DAYS
        train_d, train_w = demand[:n], weekday_all[:n]
        test_d, test_w = demand[n:], weekday_all[n:]

        # training cleanup: leading zeros, then long zero runs
        first = np.argmax(train_d > 0)
        train_d, train_w = train_d[first:], train_w[first:]
        keep = ~long_zero_mask(train_d, config.ZERO_RUN_LEN)
        train_d, train_w = train_d[keep], train_w[keep]

        # cap
        d_max = int(np.percentile(train_d, config.DEMAND_PCTL))
        train_d = np.minimum(train_d, d_max)
        test_d = np.minimum(test_d, d_max)

        # price: average over training weeks
        train_weeks = calendar["wm_yr_wk"].to_numpy()[:n]
        item_prices = prices[(prices["store_id"] == config.STORE) &
                                (prices["item_id"] == item_id) &
                                (prices["wm_yr_wk"].isin(train_weeks))]
        price = float(item_prices["sell_price"].mean())

        return {"train": (train_d, train_w), "test": (test_d, test_w),
                "price": price, "d_max": d_max}

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)

    for level, item_id in config.ITEMS.items():
        info = load_item(item_id)
        train_d, train_w = info["train"]
        test_d, test_w = info["test"]

        m0 = fit_m0(train_d, info["d_max"])
        m1 = fit_m1(train_d, train_w, info["d_max"])

        np.savez(f"{config.PROCESSED_DIR}/{level}.npz",
                    train_d=train_d, train_w=train_w,
                    test_d=test_d, test_w=test_w,
                    price=info["price"], d_max=info["d_max"],
                    m0=m0, m1=m1)

        print(level, item_id, "train days:", len(train_d),
                "| M0 rows sum to 1:", np.allclose(m0.sum(axis=1), 1),
                "| M1 rows sum to 1:", np.allclose(m1.sum(axis=1), 1))


def load_processed(level):
    """Load one item's cleaned data. level is "low", "medium", or "high". Everyone uses this."""
    f = np.load(f"{config.PROCESSED_DIR}/{level}.npz")
    return {
        "train": (f["train_d"], f["train_w"]),
        "test": (f["test_d"], f["test_w"]),
        "price": float(f["price"]),
        "d_max": int(f["d_max"]),
        "m0": f["m0"],
        "m1": f["m1"],
    }


if __name__ == "__main__":
    build_processed()