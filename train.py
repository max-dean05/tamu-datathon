"""
Train PPO bookmakers on pre-2023 seasons.

    python train.py --lam 0.1 --seeds 5 --timesteps 300000
    python train.py --lam 0.0 --seeds 5      # repeat for several lambdas to trace the risk frontier

Models are saved to models/ppo_lam{lam}_seed{seed}.zip; evaluate.py picks them up.
"""
import argparse
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from evaluate import TRAIN_SEASONS
from market_env import MarketConfig, TotalsMarketEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/cfb_totals.csv")
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--timesteps", type=int, default=300_000)
    args = p.parse_args()

    games = pd.read_csv(args.data)
    train = games[games.season.isin(TRAIN_SEASONS)]
    if train.empty:
        train = games[games.season < games.season.quantile(0.75)]
    cfg = MarketConfig(risk_lambda=args.lam)
    Path("models").mkdir(exist_ok=True)

    for seed in range(args.seeds):
        env = make_vec_env(lambda: TotalsMarketEnv(train, cfg), n_envs=8, seed=seed)
        model = PPO("MlpPolicy", env, seed=seed, n_steps=512, batch_size=256,
                    gamma=1.0, gae_lambda=0.95, ent_coef=0.01, verbose=0)
        model.learn(total_timesteps=args.timesteps, progress_bar=False)
        out = f"models/ppo_lam{args.lam}_seed{seed}"
        model.save(out)
        print(f"saved {out}.zip")


if __name__ == "__main__":
    main()
