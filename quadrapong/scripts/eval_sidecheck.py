"""Quick check: IPPO as Team 2 vs Random, to verify role specialization."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from src.envs.quadrapong_env import QuadrapongWrapper
from src.algos.ippo import IPPOTrainer

device = torch.device("cuda:0")
ckpt = torch.load("checkpoints/ippo/ippo_final.pt", map_location=device, weights_only=False)
actor_state = ckpt["actor"]
obs_dim = actor_state["mlp.0.weight"].shape[1]
action_dim = actor_state["mlp.4.weight"].shape[0]
hidden_dims = [actor_state["mlp.0.weight"].shape[0], actor_state["mlp.2.weight"].shape[0]]

trainer = IPPOTrainer(obs_dim=obs_dim, action_dim=action_dim, num_agents=2, hidden_dims=hidden_dims, device=device)
trainer.actor.load_state_dict(actor_state)
trainer.actor.eval()

env = QuadrapongWrapper(obs_type="ram", max_cycles=100000)

def run(team1_actor, team2_mode, name1, name2, n=50):
    agents = env.possible_agents
    t1i, t2i = [0,2], [1,3]
    wins = {name1: 0, name2: 0, "draw": 0}
    t1r, t2r = [], []
    for _ in range(n):
        obs, _ = env.reset(); done = False; step = 0
        ep_r = {a: 0.0 for a in agents}
        while not done:
            obs_b = np.stack([obs[a] for a in agents])
            obs_t = torch.tensor(obs_b, dtype=torch.float32, device=device)
            with torch.no_grad():
                a1 = team1_actor.mlp(obs_t[t1i]).argmax(-1).cpu().numpy()
            a2 = np.random.randint(0,6,size=2) if team2_mode == "random" else team2_mode.mlp(obs_t[t2i]).argmax(-1).cpu().numpy()
            actions = np.zeros(4, dtype=int)
            actions[t1i] = a1; actions[t2i] = a2
            ad = {a: int(actions[i]) for i,a in enumerate(agents)}
            obs, rewards, terms, truncs, _ = env.step(ad)
            for a in agents: ep_r[a] += rewards.get(a,0)
            done = any(terms.values()) or any(truncs.values()) or step >= 5000
            step += 1
        t1 = ep_r["first_0"]+ep_r["third_0"]
        t2 = ep_r["second_0"]+ep_r["fourth_0"]
        t1r.append(t1); t2r.append(t2)
        if t1>t2: wins[name1] += 1
        elif t2>t1: wins[name2] += 1
        else: wins["draw"] += 1
    print(f"{name1}(T1) vs {name2}(T2): T1_WR={wins[name1]/n:.0%} T2_WR={wins[name2]/n:.0%} Draw={wins['draw']/n:.0%} T1_r={np.mean(t1r):.1f} T2_r={np.mean(t2r):.1f}")

# IPPO T1 vs Random T2
run(trainer.actor, "random", "IPPO", "Random")
# IPPO T2 vs Random T1 (random is T1, IPPO is T2)
# For this we flip: random controls T1, IPPO controls T2
env2 = QuadrapongWrapper(obs_type="ram", max_cycles=100000)
agents2 = env2.possible_agents
t1i2, t2i2 = [0,2], [1,3]
wins2 = {"Random": 0, "IPPO": 0, "draw": 0}
t1r2, t2r2 = [], []
for _ in range(50):
    obs, _ = env2.reset(); done = False; step = 0
    ep_r = {a: 0.0 for a in agents2}
    while not done:
        obs_b = np.stack([obs[a] for a in agents2])
        obs_t = torch.tensor(obs_b, dtype=torch.float32, device=device)
        a1 = np.random.randint(0,6,size=2)  # Random T1
        with torch.no_grad():
            a2 = trainer.actor.mlp(obs_t[t2i2]).argmax(-1).cpu().numpy()  # IPPO T2
        actions = np.zeros(4, dtype=int)
        actions[t1i2] = a1; actions[t2i2] = a2
        ad = {a: int(actions[i]) for i,a in enumerate(agents2)}
        obs, rewards, terms, truncs, _ = env2.step(ad)
        for a in agents2: ep_r[a] += rewards.get(a,0)
        done = any(terms.values()) or any(truncs.values()) or step >= 5000
        step += 1
    t1 = ep_r["first_0"]+ep_r["third_0"]
    t2 = ep_r["second_0"]+ep_r["fourth_0"]
    t1r2.append(t1); t2r2.append(t2)
    if t1>t2: wins2["Random"] += 1
    elif t2>t1: wins2["IPPO"] += 1
    else: wins2["draw"] += 1
print(f"Random(T1) vs IPPO(T2): T1_WR={wins2['Random']/50:.0%} T2_WR={wins2['IPPO']/50:.0%} Draw={wins2['draw']/50:.0%} T1_r={np.mean(t1r2):.1f} T2_r={np.mean(t2r2):.1f}")
env.close(); env2.close()
