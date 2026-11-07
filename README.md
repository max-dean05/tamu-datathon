# RL Bookmaker: College Football Totals

Does a profit-maximizing RL bookmaker rediscover Levitt (2004), shading lines toward
bettor bias instead of balancing its book?

## Setup
```bash
pip install -r requirements.txt
export CFBD_API_KEY='your-key'        # free key: https://collegefootballdata.com/key
```

## Run order
```bash
python fetch_data.py --synthetic      # optional: test everything before your key arrives
python fetch_data.py --start 2014 --end 2025
python evaluate.py --sanity           # simulator sanity checks
python evaluate.py                    # baselines on test seasons (2023+)
python train.py --lam 0.1 --seeds 5   # PPO; repeat for several lambdas
python evaluate.py --lam 0.1          # baselines + trained models, with Levitt table
python evaluate.py --lam 0.1 --sharp-mult 2   # distribution-shift test
```

## Files
- `fetch_data.py`: pulls games and betting totals, caches raw JSON, builds market-profile features
- `market_env.py`: bettor agents (recreational, sharp, steam) and the Gymnasium environment
- `evaluate.py`: baselines, common-random-number evaluation, bootstrap CIs, Levitt analysis
- `train.py`: PPO training across seeds
