"""
Simulated college football totals market and a Gymnasium environment for an RL bookmaker.

One episode = one real game. Betting runs over n_steps time steps from open to kickoff.
Each step the bookmaker moves the total (or not), then bettors arrive and bet over/under
at the current line. Each bet is settled at the line it was placed at, at -110 odds.

The "fair" total is the real consensus closing line; the outcome is the real final score.
"""
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from scipy.stats import norm

PAYOUT = 100 / 110                    # a winning bettor's profit per $1 risked at -110
MOVES = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
GRID = np.arange(0, 161)              # possible final totals for computing P&L distributions
LARGE_BET = 1000.0                    # stake size that counts as a "large" bet in observations


@dataclass
class MarketConfig:
    n_steps: int = 20
    # Recreational bettors: many, small, late, biased toward overs and big-brand games
    rec_rate: float = 30.0            # expected bets per step in a max-profile game
    rec_stake_mean: float = 60.0
    rec_over_bias: float = 2.5        # points added to their estimate of the total
    rec_brand_bias: float = 1.5       # extra over bias when a brand program plays
    rec_noise: float = 6.0            # spread of individual opinions (points)
    rec_sensitivity: float = 0.15     # how strongly they respond to (estimate - line)
    # Sharps: few, large, early, accurate; only bet when the line is off
    sharp_rate: float = 0.5
    sharp_stake_mean: float = 2500.0
    sharp_noise: float = 1.0
    sharp_threshold: float = 1.0
    # Steam chasers: bet in the direction the line just moved
    steam_rate: float = 4.0
    steam_stake_mean: float = 300.0
    # Opening-line error (only used when a game has no real opening line)
    open_sd_low: float = 3.0          # low-profile games open less accurately
    open_sd_high: float = 1.5
    use_real_open: bool = True
    max_drift: float = 7.0            # max distance the line can move from the open
    # Objective: expected P&L minus risk_lambda * SD of P&L, under the fair outcome distribution
    risk_lambda: float = 0.1
    reward_scale: float = 1000.0
    outcome_sd: float | None = None   # estimated from data when None


class TotalsMarketEnv(gym.Env):
    metadata = {"render_modes": []}
    OBS_NAMES = [
        "line_minus_open", "handle_imbalance", "log_handle", "last_imbalance",
        "last_large_imbalance", "last_log_avg_stake", "time_frac", "profile",
        "brand", "weekday", "last_move",
    ]

    def __init__(self, games: pd.DataFrame, config: MarketConfig | None = None):
        super().__init__()
        self.games = games.reset_index(drop=True)
        self.cfg = config or MarketConfig()
        sd = self.cfg.outcome_sd
        if sd is None:
            sd = float((self.games.actual_total - self.games.fair_total).std())
            if not np.isfinite(sd) or sd <= 0:
                sd = 15.0
        self.outcome_sd = sd
        self.action_space = spaces.Discrete(len(MOVES))
        self.observation_space = spaces.Box(-10, 10, shape=(len(self.OBS_NAMES),), dtype=np.float32)

    # ---------- episode setup ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        idx = (options or {}).get("game_idx")
        if idx is None:
            idx = int(rng.integers(len(self.games)))
        self.game = self.games.iloc[idx]
        self.fair = float(self.game.fair_total)
        self.profile = float(self.game.profile)

        open_line = self.game.get("open_total", np.nan)
        if not (self.cfg.use_real_open and pd.notna(open_line)):
            sd = self.cfg.open_sd_low + (self.cfg.open_sd_high - self.cfg.open_sd_low) * self.profile
            open_line = self.fair + rng.normal(0, sd)
        self.open = float(np.round(open_line * 2) / 2)
        self.line = self.open

        # Discretized outcome distribution around the fair total
        cdf = norm.cdf((np.append(GRID, GRID[-1] + 1) - 0.5 - self.fair) / self.outcome_sd)
        self.probs = np.diff(cdf)
        self.probs /= self.probs.sum()

        self.t = 0
        self.pnl_vec = np.zeros(len(GRID))  # book P&L for each possible final total
        self.over_handle = self.under_handle = 0.0
        self.last = {"imb": 0.0, "large_imb": 0.0, "avg_stake": 0.0, "move": 0.0}
        self.prev_obj = 0.0
        return self._obs(), {}

    # ---------- bettor arrivals ----------
    def _bets(self):
        """Generate this step's bets: arrays of side (+1 over / -1 under) and stake."""
        c, rng, p = self.cfg, self.np_random, self.profile
        frac = self.t / max(c.n_steps - 1, 1)
        sides, stakes = [], []

        # Recreational: volume scales with game profile, ramps up toward kickoff
        n = rng.poisson(c.rec_rate * (0.15 + 0.85 * p) * 2 * (0.1 + 0.9 * frac))
        if n:
            est = (self.fair + c.rec_over_bias + c.rec_brand_bias * self.game.brand
                   + rng.normal(0, c.rec_noise, n))
            p_over = 1 / (1 + np.exp(-c.rec_sensitivity * (est - self.line)))
            sides.append(np.where(rng.random(n) < p_over, 1, -1))
            stakes.append(rng.lognormal(np.log(c.rec_stake_mean), 0.8, n))

        # Sharps: early, accurate; limits are lower in low-profile games
        n = rng.poisson(c.sharp_rate * 2 * (1 - 0.9 * frac))
        if n:
            edge = self.fair + rng.normal(0, c.sharp_noise, n) - self.line
            act = np.abs(edge) > c.sharp_threshold
            if act.any():
                sides.append(np.sign(edge[act]).astype(int))
                limit = 0.4 + 0.6 * p
                stakes.append(rng.lognormal(np.log(c.sharp_stake_mean * limit), 0.4, act.sum()))

        # Steam chasers: follow the last line move
        if self.last["move"] != 0:
            n = rng.poisson(c.steam_rate)
            if n:
                sides.append(np.full(n, int(np.sign(self.last["move"]))))
                stakes.append(rng.lognormal(np.log(c.steam_stake_mean), 0.6, n))

        if not sides:
            return np.zeros(0, int), np.zeros(0)
        return np.concatenate(sides), np.concatenate(stakes)

    # ---------- core loop ----------
    def step(self, action):
        move = float(MOVES[int(action)])
        new_line = np.clip(self.line + move, self.open - self.cfg.max_drift, self.open + self.cfg.max_drift)
        self.last["move"] = new_line - self.line
        self.line = float(new_line)

        sides, stakes = self._bets()
        if len(sides):
            # Settle each bet at the current line for every possible outcome
            diff = sides[:, None] * (GRID[None, :] - self.line)
            book = np.where(diff > 0, -stakes[:, None] * PAYOUT, np.where(diff < 0, stakes[:, None], 0.0))
            self.pnl_vec += book.sum(0)
            over = stakes[sides > 0].sum()
            under = stakes[sides < 0].sum()
            self.over_handle += over
            self.under_handle += under
            large = stakes >= LARGE_BET
            self.last.update(
                imb=(over - under) / (over + under),
                large_imb=(stakes[large & (sides > 0)].sum() - stakes[large & (sides < 0)].sum())
                          / (over + under),
                avg_stake=float(stakes.mean()),
            )
        else:
            self.last.update(imb=0.0, large_imb=0.0, avg_stake=0.0)

        self.t += 1
        obj = self._objective()
        reward = (obj - self.prev_obj) / self.cfg.reward_scale   # dense; sums to final objective
        self.prev_obj = obj
        terminated = self.t >= self.cfg.n_steps
        info = self._final_info() if terminated else {}
        return self._obs(), float(reward), terminated, False, info

    def _objective(self):
        mean = self.probs @ self.pnl_vec
        sd = np.sqrt(self.probs @ (self.pnl_vec - mean) ** 2)
        return mean - self.cfg.risk_lambda * sd

    def _obs(self):
        handle = self.over_handle + self.under_handle
        return np.array([
            (self.line - self.open) / self.cfg.max_drift,
            (self.over_handle - self.under_handle) / (handle + 1.0),
            np.log1p(handle) / 10,
            self.last["imb"],
            self.last["large_imb"],
            np.log1p(self.last["avg_stake"]) / 8,
            self.t / self.cfg.n_steps,
            self.profile,
            float(self.game.brand),
            float(self.game.weekday),
            self.last["move"],
        ], dtype=np.float32)

    def _final_info(self):
        actual = int(np.clip(self.game.actual_total, GRID[0], GRID[-1]))
        mean = float(self.probs @ self.pnl_vec)
        handle = self.over_handle + self.under_handle
        return {
            "game_id": self.game.get("game_id", None),
            "season": self.game.get("season", None),
            "profile": self.profile,
            "brand": int(self.game.brand),
            "fair": self.fair,
            "open": self.open,
            "final_line": self.line,
            "handle": handle,
            "final_imbalance": (self.over_handle - self.under_handle) / (handle + 1.0),
            "expected_pnl": mean,
            "pnl_sd": float(np.sqrt(self.probs @ (self.pnl_vec - mean) ** 2)),
            "realized_pnl": float(self.pnl_vec[actual]),
        }
