# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

RL bookmaker for college football totals: tests whether a profit-maximizing PPO agent rediscovers Levitt (2004), shading lines toward bettor bias rather than balancing the book. No test suite or linter; `evaluate.py --sanity` is the simulator check.

## Commands

```bash
pip install -r requirements.txt
export CFBD_API_KEY='...'                      # real data (collegefootballdata.com/key)
python fetch_data.py --synthetic               # fake data, no key needed
python fetch_data.py --start 2014 --end 2025   # real data -> data/cfb_totals.csv
python evaluate.py --sanity                    # simulator sanity checks
python evaluate.py [--lam 0.1] [--sharp-mult 2] [--seeds 3]   # baselines + any trained models
python train.py --lam 0.1 --seeds 5 [--timesteps 300000]      # PPO -> models/ppo_lam{lam}_seed{n}.zip
```

There is no single-test command. To iterate quickly, use `--synthetic` data and a small `--timesteps`/`--seeds`.

## Architecture

Pipeline: `fetch_data.py` → `data/cfb_totals.csv` → `market_env.py` (simulator) → `train.py` / `evaluate.py`.

- **One episode = one real game.** The fair total is the real consensus closing line and the outcome is the real final score; only the bettor flow is simulated (`MarketConfig` holds all bettor parameters: recreational, sharp, steam agents). Bets settle at the line when placed at -110. The bookmaker's action is one of `MOVES` (±1, ±0.5, 0 points) per step. Reward is expected P&L minus `risk_lambda` × SD of P&L under the fair outcome distribution.
- **Train/test split by season** lives in `evaluate.py` (`TRAIN_SEASONS`, `TEST_SEASONS`; test is 2023+). `train.py` imports `TRAIN_SEASONS` from `evaluate.py`. If the data has no test seasons (synthetic), evaluate falls back to the top 25% of seasons.
- **Model discovery by filename:** `evaluate.py` globs `models/ppo_lam{lam}_seed*.zip`, so `--lam` must match between training and evaluating (float formatting included). Baselines (`static`, `balancer`, `sharp_follower`, `random`) are plain functions of the observation vector; observation indices come from `TotalsMarketEnv.OBS_NAMES`, so changing observations means updating the baselines too.
- **Fair comparison:** `evaluate.py` runs every policy with common random numbers (same bettor seeds), bootstrap CIs, and a Levitt table (final line − fair line by game profile). `--sharp-mult` scales `sharp_rate` for the distribution-shift test.

`data/`, `models/`, `.env` are gitignored; `results/` and `figures/` are output directories (currently empty).

## Workflow

Always use the `ponytail` skill (invoke it via the Skill tool) whenever writing or changing code in this repo.
