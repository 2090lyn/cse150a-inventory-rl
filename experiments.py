import argparse
import csv
import time
from pathlib import Path
import numpy as np

import config
import baseline
import data
import evaluate
import mdp
import q_learning
import value_iteration

OUT = Path("results/experiments")
FIG = Path("figures")
TO_TEST = ["baseline", "VI-M0", "VI-M1", "Q-default", "Q-tuned"]

def read_alpha(level):
    path = Path(f"results/q_learning/{level}_full_seed0_tuning.csv")
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python q_learning.py --tune` first")
    with path.open() as f:
        return float(next(csv.DictReader(f))["alpha"])

def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - start

def run_methods(level, info, train_d, train_w, costs, seeds, passes):
    test_d, test_w = info["test"]
    d_max = info["d_max"]
    rows = []

    def add(method, policy, seconds, seed=None, **extra):
        scores = evaluate.evaluate(policy, test_d, test_w, costs)
        rows.append(dict(level=level, method=method, seed=seed, train_days=len(train_d),
                         train_seconds=round(seconds, 3), **extra, **scores))

    # baseline: best (r, S) reorder rule on the training data
    (base_policy, r, S), secs = timed(baseline.order_up_to, train_d, train_w, costs)
    add("baseline", base_policy, secs, r=r, S=S)

    # value iteration with demand models refit on this training data
    m0 = data.fit_m0(train_d, d_max)
    (vi_m0_policy, _, m0_iters), secs = timed(value_iteration.value_iteration, m0, costs)
    add("VI-M0", vi_m0_policy, secs, vi_iters=m0_iters)

    m1 = data.fit_m1(train_d, train_w, d_max)
    (vi_m1_policy, _, m1_iters), secs = timed(value_iteration.value_iteration, m1, costs)
    add("VI-M1", vi_m1_policy, secs, vi_iters=m1_iters)

    # Q-learning is random, so run several seeds; default alpha and tuned alpha
    alpha = read_alpha(level)
    for seed in seeds:
        q_default_policy, secs = timed(q_learning.q_learning, train_d, train_w, costs,
                                       seed=seed, passes=passes)
        add("Q-default", q_default_policy, secs, seed=seed, alpha=config.Q_ALPHA)

        q_tuned_policy, secs = timed(q_learning.q_learning, train_d, train_w, costs,
                                     alpha=alpha, seed=seed, passes=passes)
        add("Q-tuned", q_tuned_policy, secs, seed=seed, alpha=alpha)

    return rows

def exp_main(levels, seeds, passes):
    rows = []
    for level in levels:
        info = data.load_processed(level)
        costs = mdp.make_costs(info["price"], info["d_max"])
        print(f"[main] {level}")
        rows += run_methods(level, info, *info["train"], costs, seeds, passes)
    save(rows, "main")
    return rows

def exp_data(levels, seeds, passes):
    rows = []
    for level in levels:
        info = data.load_processed(level)
        costs = mdp.make_costs(info["price"], info["d_max"])
        d, w = info["train"]
        for n in config.DATA_SIZES:
            print(f"[data] {level} last {n}")
            rows += run_methods(level, info, d[-n:], w[-n:], costs, seeds, passes)
    save(rows, "data")
    return rows

def exp_cost(levels, seeds, passes):
    rows = []
    for level in levels:
        info = data.load_processed(level)
        for param, values in config.COST_GRID.items():
            for v in values:
                costs = mdp.make_costs(info["price"], info["d_max"], **{param: v})
                print(f"[cost] {level} {param}={v}")
                rows += [dict(r, param=param, value=v) for r in
                         run_methods(level, info, *info["train"], costs, seeds, passes)]
    save(rows, "cost")
    return rows

def exp_policy(levels, passes):
    for level in levels:
        info = data.load_processed(level)
        costs = mdp.make_costs(info["price"], info["d_max"])
        d, w = info["train"]
        print(f"[policy] {level}")
        _, r, S = baseline.order_up_to(d, w, costs)
        vi, _, _ = value_iteration.value_iteration(info["m1"], costs)
        alpha = read_alpha(level)
        q, details = q_learning.q_learning(d, w, costs, alpha=alpha, passes=passes, seed=0, return_details=True)
        np.savez(OUT / f"policies_{level}.npz", baseline=baseline.reorder_policy(r, S, costs["i_max"]),
                 vi_m1=vi, q_tuned=q, q_visited=details["visits"].sum(axis=2) > 0, r=r, S=S)


def save(rows, name):
    OUT.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with (OUT / f"{name}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {OUT / name}.csv")

def load(name):
    with (OUT / f"{name}.csv").open() as f:
        return list(csv.DictReader(f))

def summarize(rows, group_keys, metric="profit"):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in group_keys), []).append(float(r[metric]))
    return {k: (np.mean(v), np.std(v)) for k, v in groups.items()}


def print_main_table(rows):
    metrics = ["profit", "lost_units", "stockout_days", "fill_rate", "orders", "train_seconds"]
    sums = {m: summarize(rows, ["level", "method"], m) for m in metrics}
    print(f"\n{'item':7s} {'method':10s} " + " ".join(f"{m:>16s}" for m in metrics))
    for (level, method), _ in sums["profit"].items():
        cells = []
        for m in metrics:
            mean, std = sums[m][(level, method)]
            cells.append(f"{mean:9.2f} ± {std:5.2f}" if std else f"{mean:16.2f}")
        print(f"{level:7s} {method:10s} " + " ".join(cells))

ITEM_LABEL = {"low": "Low-demand item (~2/day)", "medium": "Medium-demand item (~8/day)",
              "high": "High-demand item (~19/day)"}
COST_LABEL = {"order_frac": ("Fixed cost per order, K", "K = 2× price"),
              "lost_frac": ("Penalty per lost sale, g (on top of lost revenue)", "g = 0.2× price")}
PROFIT_Y = "Total test profit ($)\nhigher = better"


def plot_all():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(exist_ok=True)
    levels = ["low", "medium", "high"]
    colors = dict(zip(TO_TEST, ["#7f7f7f", "#1f77b4", "#17becf", "#ff7f0e", "#d62728"]))
    note = "Each policy run on the 141 held-out test days (days 1801-1941), starting at 0 inventory."

    # 1. main comparison
    if (OUT / "main.csv").exists():
        rows = load("main")
        print_main_table(rows)
        for metric, label, title in [
                ("profit", PROFIT_Y, "Which method makes the most money?"),
                ("stockout_days", "Days with lost sales\n(out of 141, lower = better)",
                 "How often does each method run out of stock?")]:
            s = summarize(rows, ["level", "method"], metric)
            fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
            for ax, level in zip(axes, levels):
                means = [s.get((level, m), (np.nan, 0))[0] for m in TO_TEST]
                stds = [s.get((level, m), (0, 0))[1] for m in TO_TEST]
                bars = ax.bar(TO_TEST, means, yerr=stds, capsize=3, color=[colors[m] for m in TO_TEST])
                ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
                ax.set_title(ITEM_LABEL[level])
                ax.tick_params(axis="x", rotation=35)
                ax.grid(axis="y", alpha=.3)
                ax.margins(y=.12)
            axes[0].set_ylabel(label)
            fig.suptitle(f"{title}\n{note} Q-learning: mean ± std over seeds.", fontsize=10)
            fig.savefig(FIG / f"results_main_{metric}.png", dpi=160)
            plt.close(fig)

    # 2. less data
    if (OUT / "data.csv").exists():
        s = summarize(load("data"), ["level", "method", "train_days"])
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
        for ax, level in zip(axes, levels):
            for m in TO_TEST:
                xs = [n for n in config.DATA_SIZES if (level, m, str(n)) in s]
                if not xs:
                    continue
                mean = np.array([s[(level, m, str(n))][0] for n in xs])
                std = np.array([s[(level, m, str(n))][1] for n in xs])
                ax.plot(xs, mean, "o-", label=m, color=colors[m])
                ax.fill_between(xs, mean - std, mean + std, alpha=.15, color=colors[m])
            ax.set(xscale="log", title=ITEM_LABEL[level], xticks=config.DATA_SIZES,
                   xticklabels=config.DATA_SIZES, xlabel="Training days used (most recent N)")
            ax.grid(alpha=.3)
        axes[0].set_ylabel(PROFIT_Y)
        axes[0].legend(fontsize=8)
        fig.suptitle("How much training data does each method need?\n"
                     f"Each method trained on only the last N training days, then tested the same way. {note}",
                     fontsize=10)
        fig.savefig(FIG / "results_less_data.png", dpi=160)
        plt.close(fig)

    # 3. cost sensitivity
    if (OUT / "cost.csv").exists():
        s = summarize(load("cost"), ["level", "method", "param", "value"])
        params = list(config.COST_GRID)
        fig, axes = plt.subplots(len(params), 3, figsize=(12, 3.8 * len(params)), constrained_layout=True)
        for i, param in enumerate(params):
            name, default = COST_LABEL[param]
            for j, level in enumerate(levels):
                ax = axes[i, j]
                for m in TO_TEST:
                    xs = [v for v in config.COST_GRID[param] if (level, m, param, str(v)) in s]
                    if not xs:
                        continue
                    ax.plot(xs, [s[(level, m, param, str(v))][0] for v in xs], "o-", label=m, color=colors[m])
                dflt = config.ORDER_COST_FRAC if param == "order_frac" else config.LOST_FRAC
                ax.axvline(dflt, color="k", ls=":", lw=1)
                ax.set(title=ITEM_LABEL[level], xlabel=f"{name}\n(× item price; dotted line = our default, {default})")
                ax.grid(alpha=.3)
            axes[i, 0].set_ylabel(PROFIT_Y)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("Do the results depend on our assumed costs?\n"
                     "One cost changed at a time (others at defaults); every method re-trained under the new costs. "
                     + note, fontsize=10)
        fig.savefig(FIG / "results_cost_sensitivity.png", dpi=160)
        plt.close(fig)

    # 4. policy heatmaps
    files = [OUT / f"policies_{lv}.npz" for lv in levels]
    if all(f.exists() for f in files):
        days = ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"]
        cols = [("baseline", "Baseline (reorder rule)"), ("vi_m1", "Value iteration (M1)"),
                ("q_tuned", "Q-learning (tuned, seed 0)")]
        fig, axes = plt.subplots(3, 3, figsize=(11, 12), constrained_layout=True)
        for i, (level, f) in enumerate(zip(levels, files)):
            P = np.load(f)
            vmax = max(P[k].max() for k, _ in cols)
            for j, (key, title) in enumerate(cols):
                ax = axes[i, j]
                img = np.ma.masked_array(P[key], mask=~P["q_visited"]) if key == "q_tuned" else P[key]
                cmap = plt.get_cmap("viridis").copy()
                cmap.set_bad("#dddddd")
                im = ax.imshow(img, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
                ax.set(xticks=range(7), xticklabels=days, title=f"{title}" if i == 0 else None)
                if j == 0:
                    ax.set_ylabel(f"{ITEM_LABEL[level]}\ninventory at start of day")
            fig.colorbar(im, ax=axes[i, :], label="units ordered", shrink=.9)
        fig.suptitle("What does each learned policy actually do?\n"
                     "Color = how many units to order given today's inventory (y) and weekday (x). "
                     "Grey = states Q-learning never visited in training.", fontsize=10)
        fig.savefig(FIG / "results_policies.png", dpi=160)
        plt.close(fig)
    print(f"figures saved to {FIG}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("which", choices=["main", "data", "cost", "policy", "all", "plot"])
    p.add_argument("--levels", nargs="+", default=["low", "medium", "high"])
    p.add_argument("--seeds", type=int, default=3, help="number of Q-learning seeds")
    p.add_argument("--passes", type=int, default=config.Q_PASSES)
    a = p.parse_args()
    seeds = list(range(a.seeds))
    if a.which in ("main", "all"):
        exp_main(a.levels, seeds, a.passes)
    if a.which in ("data", "all"):
        exp_data(a.levels, seeds, a.passes)
    if a.which in ("cost", "all"):
        exp_cost(a.levels, seeds, a.passes)
    if a.which in ("policy", "all"):
        exp_policy(a.levels, a.passes)
    plot_all()
