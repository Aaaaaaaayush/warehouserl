"""
verify_env.py — Step 1 smoke test.
Verifies CUDA, all module imports, env reset, and network instantiation.
Delete after Step 1 is confirmed complete.
"""
import sys
sys.path.insert(0, ".")

import torch
import pettingzoo
import gymnasium
import mlflow
import fastapi

print("=== WarehouseRL — Step 1 Verification ===\n")

# CUDA
print(f"PyTorch:        {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version:   {torch.version.cuda}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU:            {props.name}")
    print(f"VRAM:           {props.total_memory / 1e9:.1f} GB")
print(f"PettingZoo:     {pettingzoo.__version__}")
print(f"Gymnasium:      {gymnasium.__version__}")
print(f"MLflow:         {mlflow.__version__}")
print(f"FastAPI:        {fastapi.__version__}")

print("\n=== Module Imports ===\n")

from src.environment.config_loader import load_config, ConfigError
print("config_loader   OK")

from src.environment.warehouse_env import WarehouseEnv, CellType, Action
print("warehouse_env   OK")

from src.agents.q_network import QNetwork
print("q_network       OK")

from src.agents.qmix import QMixingNetwork
print("qmix            OK")

from src.agents.replay_buffer import ReplayBuffer
print("replay_buffer   OK")

from src.agents.comm_hooks import pre_act_comm, post_act_comm
print("comm_hooks      OK")

print("\n=== Config Load — Scenario 1 ===\n")

cfg = load_config(1)
assert cfg.scenario.id == 1
assert cfg.agents.count == 4
assert cfg.grid.width == 8
print(f"Scenario:       {cfg.scenario.id} — {cfg.scenario.name}")
print(f"Grid:           {cfg.grid.width}x{cfg.grid.height}")
print(f"Agents:         {cfg.agents.count}")
print(f"Episodes:       {cfg.training.total_episodes:,}")
print(f"Delivery reward:{cfg.rewards.delivery_reward}")

print("\n=== Environment Reset ===\n")

env = WarehouseEnv(cfg)
obs, infos = env.reset(seed=42)
assert len(obs) == cfg.agents.count, f"Expected {cfg.agents.count} obs, got {len(obs)}"
sample_obs = list(obs.values())[0]
expected_obs_size = (2 * cfg.agents.observation_radius + 1) ** 2 + 6
assert sample_obs.shape == (expected_obs_size,), f"Bad obs shape: {sample_obs.shape}"
print(f"Active agents:  {len(obs)}")
print(f"Obs shape:      {sample_obs.shape}  (expected {expected_obs_size})")
print(f"Action space:   {env.action_space('agent_0')}")

# Step with random actions
actions = {a: env.action_space(a).sample() for a in env.agents}
obs2, rew, terms, truncs, info2 = env.step(actions)
print(f"Step OK:        rewards={[round(r,2) for r in rew.values()]}")

print("\n=== Network Instantiation ===\n")

obs_size = expected_obs_size
act_size = 7
H, W = cfg.grid.height, cfg.grid.width
state_size = H * W + cfg.agents.count * 6

net = QNetwork(obs_size=obs_size, act_size=act_size)
mixer = QMixingNetwork(n_agents=cfg.agents.count, state_size=state_size)
buf = ReplayBuffer(
    capacity=500, n_agents=cfg.agents.count,
    obs_size=obs_size, state_size=state_size,
    max_ep_len=cfg.training.max_steps_per_episode,
)
q_params = sum(p.numel() for p in net.parameters())
m_params = sum(p.numel() for p in mixer.parameters())
print(f"QNetwork:       {q_params:,} parameters")
print(f"QMixNetwork:    {m_params:,} parameters")
print(f"ReplayBuffer:   capacity={buf.capacity} episodes, can_sample={buf.can_sample(1)}")

# Quick forward pass on CPU
dummy_obs = torch.zeros(1, obs_size)
dummy_h   = QNetwork.init_hidden(batch_size=1)
q_vals, new_h = net(dummy_obs, dummy_h)
assert q_vals.shape == (1, act_size)
print(f"QNetwork fwd:   output shape {tuple(q_vals.shape)}  OK")

dummy_qs    = torch.zeros(1, cfg.agents.count)
dummy_state = torch.zeros(1, state_size)
q_tot = mixer(dummy_qs, dummy_state)
assert q_tot.shape == (1, 1)
print(f"Mixer fwd:      output shape {tuple(q_tot.shape)}  OK")

# V2 comm hooks
comm = pre_act_comm({}, env.agents)
assert all(v is None for v in comm.values())
print(f"comm_hooks:     pre_act_comm returns None for all agents  OK")

print("\n=== ALL CHECKS PASSED ===")
print("Step 1: Project scaffold + venv setup COMPLETE.")
