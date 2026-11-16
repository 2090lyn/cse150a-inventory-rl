import os

import matplotlib.pyplot as plt
import numpy as np

import config
from data import load_processed

os.makedirs("figures", exist_ok=True)
LEVELS = ["low", "medium", "high"]
WEEKDAYS = ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"]

# 1. Daily demand over time, with the train/test split marked
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
for ax, level in zip(axes, LEVELS):
    info = load_processed(level)
    train_d, _ = info["train"]
    test_d, _ = info["test"]
    demand = np.concatenate([train_d, test_d])
    ax.plot(demand, linewidth=0.5)
    ax.axvline(len(train_d), color="red", linestyle="--", label="test starts")
    ax.set_title(f"{level}: {config.ITEMS[level]}")
    ax.set_ylabel("units sold")
axes[0].legend()
axes[-1].set_xlabel("day")
fig.tight_layout()
fig.savefig("figures/demand_over_time.png", dpi=200)

# 2. Demand distribution (M0) for each item
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, level in zip(axes, LEVELS):
    info = load_processed(level)
    m0 = info["m0"][0]                      # all 7 rows are the same, so take row 0
    ax.bar(np.arange(len(m0)), m0 )
    ax.set_title(level)
    ax.set_xlabel("daily demand")
axes[0].set_ylabel("probability")
fig.tight_layout()
fig.savefig("figures/demand_hist_m0.png", dpi=200)

# 3. Average demand by weekday (why M1 might help)
fig, ax = plt.subplots(figsize=(7, 4))
for level in LEVELS:
    info = load_processed(level)
    m1 = info["m1"]                         # shape (7, d_max + 1)
    values = np.arange(m1.shape[1])
    mean_by_day = m1 @ values                 # expected demand for each weekday
    ax.plot(WEEKDAYS, mean_by_day, marker="o", label=level)
ax.set_ylabel("average daily demand")
ax.legend()
fig.tight_layout()
fig.savefig("figures/demand_by_weekday.png", dpi=200)

print("saved 3 figures to figures/")