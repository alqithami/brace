from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import joblib
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class ContainmentEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, step_stats: dict, sim_model, K=10, max_steps=15,
                 B=10.0, b=2.5, costs=(1.0,0.5,2.0), effects=(0.15,0.08,0.25),
                 alpha=1.0, beta=0.6, eta=0.4, lam=0.2, xi=1.0):
        super().__init__()
        self.step_stats = step_stats
        self.sim = sim_model
        self.K = K
        self.max_steps = max_steps
        self.B = float(B); self.b = float(b)
        self.cost_debunk, self.cost_friction, self.cost_escalate = map(float, costs)
        self.eff_debunk, self.eff_friction, self.eff_escalate = map(float, effects)
        self.alpha, self.beta, self.eta, self.lam, self.xi = alpha, beta, eta, lam, xi

        self.action_space = spaces.MultiDiscrete([K, 2, 2, 2])
        self.observation_space = spaces.Box(low=-10, high=10, shape=(7,), dtype=np.float32)

        self.cascade_ids = list(step_stats.keys())
        self.reset(seed=42)

    def _get_obs(self):
        t_norm = self.t / max(1, self.max_steps)
        budget_left = self.budget_left / max(1e-6, self.B)
        hubs = self.hubs
        hub_mean = float(np.mean(hubs)) if len(hubs) else 0.0
        hub_max = float(np.max(hubs)) if len(hubs) else 0.0
        return np.array([t_norm, self.prev, self.growth, self.panic, budget_left, hub_mean, hub_max], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cid = self.np_random.choice(self.cascade_ids)
        st = self.step_stats[self.cid]
        self.t = 0
        self.prev = 0.0
        self.growth = 0.0
        self.panic = float(st.get("panic0", 0.0))
        self.hubs = st.get("hub_strengths", np.zeros(self.K, dtype=float))
        self.budget_left = self.B
        self.last_prev = 0.0
        return self._get_obs(), {}

    def step(self, action):
        target_idx, comms, soc, ir = map(int, action)
        c = comms*self.cost_debunk + soc*self.cost_friction + ir*self.cost_escalate
        violation = 1.0 if (c > self.b or c > self.budget_left) else 0.0
        if c > self.budget_left:
            comms = soc = ir = 0
            c = 0.0
        self.budget_left -= c

        x = np.array([[self.t, self.prev, self.growth, self.max_steps - self.t]], dtype=float)
        base_next = float(self.sim.predict(x)[0])
        base_next = max(0.0, base_next)

        hub_strength = float(self.hubs[target_idx]) if len(self.hubs) else 0.0
        eff = 1.0
        if comms: eff *= (1.0 - self.eff_debunk*hub_strength)
        if soc:   eff *= (1.0 - self.eff_friction*hub_strength)
        if ir:    eff *= (1.0 - self.eff_escalate*hub_strength)
        next_growth = base_next * eff

        self.last_prev = self.prev
        self.prev = min(1.0, self.prev + next_growth/100.0)
        self.growth = min(1.0, next_growth/50.0)
        self.panic = float(np.clip(self.panic - 0.06*comms + 0.02*self.growth, 0.0, 1.0))

        r = -self.alpha*self.prev - self.beta*(self.prev-self.last_prev) - self.eta*self.panic - self.lam*c - self.xi*violation
        self.t += 1
        terminated = (self.t >= self.max_steps) or (self.prev <= 0.02)
        return self._get_obs(), float(r), terminated, False, {"cost": c, "violation": violation, "prev": self.prev}

def load_ids(splits_root: Path, dataset: str, split: str, part: str):
    if split == "within":
        fp = splits_root / dataset / "within" / f"{part}.csv"
    else:
        fp = splits_root / dataset / split / f"{part}.csv"
    return pd.read_csv(fp)["cascade_id"].astype(str).tolist()

def build_step_stats(nodes: pd.DataFrame, edges: pd.DataFrame, panic: pd.DataFrame|None, ids: set[str], K=10):
    nodes = nodes[nodes["cascade_id"].isin(ids)]
    edges = edges[edges["cascade_id"].isin(ids)]
    if panic is not None:
        panic = panic[panic["cascade_id"].isin(ids)]

    step_stats = {}
    for cid, g in nodes.groupby("cascade_id"):
        e = edges[edges["cascade_id"]==cid]
        if len(e):
            outdeg = e.groupby("parent_id").size().sort_values(ascending=False)
            vals = outdeg.values.astype(float)
            if len(vals):
                vals = vals / (vals.max() + 1e-12)
                hubs = np.zeros(K, dtype=float)
                hubs[:min(K,len(vals))] = vals[:min(K,len(vals))]
            else:
                hubs = np.zeros(K, dtype=float)
        else:
            hubs = np.zeros(K, dtype=float)

        panic0 = 0.0
        if panic is not None and len(panic):
            sub = panic[panic["cascade_id"]==cid]
            if len(sub):
                panic0 = float(sub["panic"].mean())
        step_stats[str(cid)] = {"hub_strengths": hubs, "panic0": panic0}
    return step_stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["twitter15","twitter16","pheme"])
    ap.add_argument("--split", default="within")
    ap.add_argument("--part", default="train", choices=["train","dev","test"])
    ap.add_argument("--processed", default="data_processed")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--features", default="features")
    ap.add_argument("--sim-model", default=None)
    ap.add_argument("--out", default="models/rl")
    ap.add_argument("--total-timesteps", type=int, default=20000)
    args = ap.parse_args()

    processed = Path(args.processed)/args.dataset
    splits_root = Path(args.splits)
    ids = set(load_ids(splits_root, args.dataset, args.split, args.part))

    nodes = pd.read_parquet(processed/"nodes.parquet"); nodes["cascade_id"]=nodes["cascade_id"].astype(str)
    edges = pd.read_parquet(processed/"edges.parquet"); edges["cascade_id"]=edges["cascade_id"].astype(str)

    panic_path = Path(args.features)/f"panic_{args.dataset}.parquet"
    panic = pd.read_parquet(panic_path) if panic_path.exists() else None

    sim_path = Path(args.sim_model) if args.sim_model else Path("models/simulator")/args.dataset/f"sim_step_{args.split}.joblib"
    sim = joblib.load(sim_path)

    step_stats = build_step_stats(nodes, edges, panic, ids, K=10)
    env = DummyVecEnv([lambda: ContainmentEnv(step_stats, sim_model=sim, K=10, max_steps=15)])

    model = PPO("MlpPolicy", env, verbose=1, n_steps=256, batch_size=64, learning_rate=3e-4, gamma=0.99)
    model.learn(total_timesteps=int(args.total_timesteps))

    out = Path(args.out)/args.dataset
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out/f"ppo_{args.split}_{args.part}"))
    print("Saved:", out/f"ppo_{args.split}_{args.part}")

if __name__ == "__main__":
    main()
