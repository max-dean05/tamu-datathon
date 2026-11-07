"""
Baselines, evaluation harness, sanity checks, and the Levitt analysis.

    python evaluate.py --sanity            # run simulator sanity checks first
    python evaluate.py                     # baselines (+ any trained PPO models) on the test seasons
    python evaluate.py --sharp-mult 2      # distribution-shift test: twice as many sharps

Every policy plays the same games with the same bettor random numbers (common random
numbers), so differences between policies are not luck of the draw.
"""
import argparse
import dataclasses
import glob

import numpy as np
import pandas as pd

from market_env import MarketConfig, TotalsMarketEnv

# Action indices: 0=-1, 1=-0.5, 2=hold, 3=+0.5, 4=+1
OBS = {name: i for i, name in enumerate(TotalsMarketEnv.OBS_NAMES)}
TRAIN_SEASONS = range(2014, 2023)   # pre-2023 clock rule change
TEST_SEASONS = range(2023, 2027)


def static_policy(obs, rng):
    return 2


def balancer_policy(obs, rng):
    imb = obs[OBS["handle_imbalance"]]
    return 3 if imb > 0.1 else 1 if imb < -0.1 else 2


def sharp_follower_policy(obs, rng):
    big = obs[OBS["last_large_imbalance"]]
    return 4 if big > 0.2 else 0 if big < -0.2 else 2


def random_policy(obs, rng):
    return int(rng.integers(5))


BASELINES = {
    "static": static_policy,
    "balancer": balancer_policy,
    "sharp_follower": sharp_follower_policy,
    "random": random_policy,
}


def ppo_policy(path):
    from stable_baselines3 import PPO
    model = PPO.load(path)
    return lambda obs, rng: int(model.predict(obs, deterministic=True)[0])


def run_policy(policy, games, cfg, seed=0):
    env = TotalsMarketEnv(games, cfg)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(len(games)):
        obs, _ = env.reset(seed=seed * 1_000_003 + i, options={"game_idx": i})
        done = False
        while not done:
            obs, _, done, _, info = env.step(policy(obs, rng))
        rows.append(info)
    return pd.DataFrame(rows)


def bootstrap_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    means = rng.choice(x, (n, len(x))).mean(1)
    return np.percentile(means, [2.5, 97.5])


def summarize(name, df):
    lo, hi = bootstrap_ci(df.expected_pnl)
    return {
        "policy": name,
        "exp_pnl_per_game": df.expected_pnl.mean(),
        "exp_pnl_95ci": f"[{lo:.0f}, {hi:.0f}]",
        "exp_hold_%": 100 * df.expected_pnl.sum() / df.handle.sum(),
        "realized_hold_%": 100 * df.realized_pnl.sum() / df.handle.sum(),
        "pnl_sd_per_game": df.pnl_sd.mean(),
        "mean_|imbalance|": df.final_imbalance.abs().mean(),
    }


def levitt_table(name, df):
    """Does the book shade lines toward recreational (over) bias instead of balancing?"""
    df = df.assign(shade=df.final_line - df.fair,
                   tier=pd.cut(df.profile, [-0.01, 0.33, 0.66, 1.0], labels=["low", "mid", "high"]))
    t = df.groupby("tier", observed=True).agg(
        games=("shade", "size"), mean_shade_pts=("shade", "mean"),
        mean_over_imbalance=("final_imbalance", "mean"))
    t.insert(0, "policy", name)
    return t.reset_index()


def sanity_checks(games):
    base = MarketConfig()
    sample = games.sample(min(400, len(games)), random_state=0)
    checks = [
        ("Unbiased recreational only, static -> hold near 4.5%",
         dataclasses.replace(base, rec_over_bias=0, rec_brand_bias=0, sharp_rate=0, steam_rate=0,
                             use_real_open=False, open_sd_low=0, open_sd_high=0), static_policy),
        ("Sharps only, static -> should lose money",
         dataclasses.replace(base, rec_rate=0, steam_rate=0, use_real_open=False), static_policy),
    ]
    for label, cfg, pol in checks:
        df = run_policy(pol, sample, cfg)
        print(f"{label}\n   expected hold = {100 * df.expected_pnl.sum() / max(df.handle.sum(), 1):.2f}%, "
              f"mean P&L/game = {df.expected_pnl.mean():.0f}")
    full = {n: run_policy(p, sample, base).expected_pnl.mean() for n, p in BASELINES.items()}
    print("Default market, mean expected P&L/game:", {k: round(v) for k, v in full.items()})
    best = max(v for k, v in full.items() if k != "random")
    print("   random should not beat the best baseline:", full["random"] < best)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/cfb_totals.csv")
    p.add_argument("--sanity", action="store_true")
    p.add_argument("--sharp-mult", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--seeds", type=int, default=3, help="bettor-randomness seeds per policy")
    args = p.parse_args()

    games = pd.read_csv(args.data)
    if args.sanity:
        sanity_checks(games)
        return

    test = games[games.season.isin(TEST_SEASONS)]
    if test.empty:   # e.g. synthetic data with different seasons
        test = games[games.season >= games.season.quantile(0.75)]
    cfg = MarketConfig(risk_lambda=args.lam, sharp_rate=MarketConfig.sharp_rate * args.sharp_mult)
    print(f"Test games: {len(test)} | sharp multiplier: {args.sharp_mult}")

    policies = dict(BASELINES)
    for path in sorted(glob.glob(f"models/ppo_lam{args.lam}_seed*.zip")):
        policies[path.split("/")[-1].removesuffix(".zip")] = ppo_policy(path)

    summary, levitt = [], []
    for name, pol in policies.items():
        df = pd.concat([run_policy(pol, test, cfg, seed=s) for s in range(args.seeds)])
        summary.append(summarize(name, df))
        levitt.append(levitt_table(name, df))

    pd.set_option("display.width", 160)
    print("\n=== Performance ===")
    print(pd.DataFrame(summary).round(2).to_string(index=False))
    print("\n=== Levitt analysis: line shading (final line - fair) by game profile ===")
    print(pd.concat(levitt).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
