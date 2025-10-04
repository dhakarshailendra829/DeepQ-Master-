import streamlit as st
import torch
import numpy as np
import pandas as pd
import time
import io
import matplotlib.pyplot as plt
import seaborn as sns
import os
from core.env import GridEnv
from core.agent import DQNAgent
from core.utils import plot_grid, plot_rewards

# Page Config & Basic Styling

st.set_page_config(
    page_title="DQN Grid Navigator — Ultimate Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg,#0f172a 0%, #071033 100%); color: #E6EEF8; }
    .big-title { font-size:36px; font-weight:800; text-align:center;
      background: linear-gradient(90deg,#FFD166,#06D6A0,#118AB2);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .card { background: rgba(255,255,255,0.05); padding: 14px; border-radius: 10px; }
    .quote { font-size:18px; text-align:center; font-style:italic; margin-top:20px; }
    .author { font-weight:600; background: linear-gradient(90deg,#FFD166,#06D6A0,#118AB2);
              -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="big-title">🧭 DQN Grid Navigator — Advanced Dashboard</div>', unsafe_allow_html=True)

st.write("Professional RL dashboard with Seaborn analytics, interactive controls, gradient paths, episode replay and more.")

# Sidebar Controls
st.sidebar.header("⚙️ Environment & Model")
grid_size = st.sidebar.slider("Grid Size", 5, 20, 10)
obstacle_prob = st.sidebar.slider("Obstacle Probability", 0.0, 0.5, 0.1, step=0.01)
max_steps = st.sidebar.slider("Max Steps / Episode", 20, 300, 100)
OBS_WINDOW = 7
st.sidebar.info(f"Observation Window fixed: {OBS_WINDOW}")

# Theme selector
theme_choice = st.sidebar.selectbox("Seaborn Theme", ["darkgrid", "whitegrid", "dark", "white", "ticks"])
palette_choice = st.sidebar.selectbox("Palette", ["husl", "rocket", "mako", "viridis", "coolwarm"])
sns.set_theme(style=theme_choice)
sns.set_palette(palette_choice)

st.sidebar.header("🕹 Simulation")
auto_speed = st.sidebar.slider("Auto-run speed (sec)", 0.01, 0.5, 0.08, step=0.01)
batch_episodes = st.sidebar.number_input("Batch Eval Episodes", min_value=1, max_value=1000, value=100, step=10)

# Load Model
env = GridEnv(size=grid_size, obstacle_prob=obstacle_prob, obs_window=OBS_WINDOW, max_steps=max_steps)
state_dim = env.obs_window * env.obs_window + 2
agent = DQNAgent(state_dim, 5)

model_path = "models/dqn_navigation.pth"
model_loaded = False
try:
    agent.q_net.load_state_dict(torch.load(model_path, map_location='cpu'))
    agent.q_net.eval()
    model_loaded = True
    st.sidebar.success(f"Model Loaded: {model_path}")
except Exception:
    st.sidebar.warning("Model not found — place at models/dqn_navigation.pth")

# Top Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.metric("Total Episodes (trained)", "3000")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    try:
        rewards_df = pd.read_csv("data/training_rewards.csv")
        avg100 = rewards_df['reward'].tail(100).mean()
        st.metric("Avg Reward (last 100)", f"{avg100:.2f}")
    except Exception:
        st.metric("Avg Reward (last 100)", "N/A")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.metric("Expected Success Rate", "~80–95%")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.metric("Grid / Obst / ObsWnd", f"{grid_size}/{obstacle_prob:.2f}/{OBS_WINDOW}")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# LIVE SIMULATION PANEL
left, right = st.columns([1.2, 1])
with left:
    st.subheader("Live Agent Simulation")
    sim_placeholder = st.empty()

    if "sim_state" not in st.session_state:
        st.session_state.sim_state = None

    if st.button("♻️ Reset Simulation"):
        st.session_state.sim_state = None

    if st.session_state.sim_state is None:
        sim_env = GridEnv(size=grid_size, obstacle_prob=obstacle_prob, obs_window=OBS_WINDOW, max_steps=max_steps)
        state = sim_env.reset()
        st.session_state.sim_state = {"env": sim_env, "state": state, "positions": [tuple(map(int, sim_env.agent_pos))], "steps": 0, "done": False}

    sim_env = st.session_state.sim_state["env"]
    sim_state = st.session_state.sim_state

    col_a, col_b, col_c = st.columns(3)
    step_once = col_a.button("Next Step")
    auto_run = col_b.button("Auto Run")
    stop_run = col_c.button("Stop")

# live state info 
    with st.expander("Agent Live State"):
        ar, ac = map(int, sim_env.agent_pos)
        gr, gc = map(int, sim_env.goal)
        dx, dy = int(gr - ar), int(gc - ac)
        st.write(f"**Agent Pos:** ({ar},{ac}) | **Goal:** ({gr},{gc})")
        st.write(f"**Δx:** {dx}, **Δy:** {dy} | **Steps:** {sim_state['steps']}")

# single step
    if step_once and not sim_state["done"]:
        action = agent.select_action(sim_state["state"], epsilon=0.0)
        ns, r, d, _ = sim_env.step(action)
        sim_state["state"] = ns
        sim_state["positions"].append(tuple(map(int, sim_env.agent_pos)))
        sim_state["steps"] += 1
        sim_state["done"] = d

# auto run
    if auto_run:
        for _ in range(max_steps):
            if sim_state["done"]:
                break
            action = agent.select_action(sim_state["state"], epsilon=0.0)
            ns, r, d, _ = sim_env.step(action)
            sim_state["state"] = ns
            sim_state["positions"].append(tuple(map(int, sim_env.agent_pos)))
            sim_state["steps"] += 1
            sim_state["done"] = d
            with sim_placeholder.container():
                plot_grid(sim_env, sim_state["positions"], figsize=(5,5))
                st.pyplot(plt.gcf())
                plt.clf()
            time.sleep(auto_speed)
            if stop_run:
                break

    if not auto_run:
        with sim_placeholder.container():
            plot_grid(sim_env, sim_state["positions"], figsize=(5,5))
            st.pyplot(plt.gcf())
            plt.clf()

# When episode ends: show efficiency metrics & readable positions
    if sim_state["done"]:
        ar, ac = map(int, sim_env.agent_pos)
        gr, gc = map(int, sim_env.goal)
        if sim_env.agent_pos == sim_env.goal:
            st.success("🎉 Agent reached the goal!")
            st.balloons()
        else:
            st.error("Episode ended without success.")

# Path efficiency: agent steps vs Manhattan distance 
        path_len = len(sim_state["positions"]) - 1
        manhattan = abs(gr - sim_state["positions"][0][0]) + abs(gc - sim_state["positions"][0][1])
# Better lower bound: Manhattan between start and goal
        start = tuple(map(int, sim_env.start))
        manhattan = abs(gr - start[0]) + abs(gc - start[1])
        efficiency = path_len / max(1, manhattan)
        st.write(f"Path length (agent): {path_len}  |  Manhattan lower bound: {manhattan}  |  Efficiency (agent/manhattan): {efficiency:.2f}")

# Training Analytics
with right:
    st.subheader("Seaborn Training Analytics")
    try:
        rewards_df = pd.read_csv("data/training_rewards.csv")
        rewards = rewards_df['reward'].values

        fig1, ax1 = plt.subplots(figsize=(8,3))
        sns.lineplot(x=np.arange(len(rewards)), y=rewards, ax=ax1)
        if len(rewards) >= 100:
            ma = np.convolve(rewards, np.ones(100)/100, mode='valid')
            ax1.plot(np.arange(99, len(rewards)), ma, color='orange', label='100-episode MA')
        ax1.legend(); ax1.set_title("Reward Curve")
        st.pyplot(fig1)
        plt.clf()

        fig2, (ax2, ax3) = plt.subplots(1,2, figsize=(8,3))
        sns.histplot(rewards, kde=True, ax=ax2, color='#06D6A0')
        sns.boxplot(x=rewards, ax=ax3, palette="coolwarm")
        ax2.set_title("Distribution"); ax3.set_title("Boxplot")
        st.pyplot(fig2)
        plt.clf()
    except Exception:
        st.info("Training rewards not found at data/training_rewards.csv")

# BATCH EVALUATION
st.markdown("---")
st.subheader("Batch Evaluation & Replay")

if st.button("Run Batch Evaluation"):
    if not model_loaded:
        st.warning("Load the model first.")
    else:
        summary = []
        visit_counts = np.zeros((grid_size, grid_size), dtype=int)
        for ep in range(batch_episodes):
            s = env.reset(); done = False; steps = 0; total_r = 0
            traj = [tuple(map(int, env.agent_pos))]
            while not done and steps < max_steps:
                a = agent.select_action(s, epsilon=0.0)
                s, r, done, _ = env.step(a)
                total_r += r
                steps += 1
                traj.append(tuple(map(int, env.agent_pos)))
# safe increment using tuple indices
                pr, pc = traj[-1]
                visit_counts[pr, pc] += 1
            summary.append({"episode":ep+1,"steps":steps,"reward":total_r,"success":int(env.agent_pos==env.goal),"trajectory":traj})

        df = pd.DataFrame(summary)
        st.dataframe(df.tail(10), use_container_width=True)
        st.metric("Batch Success Rate", f"{df['success'].mean()*100:.2f}%")

# heatmap controls
        norm_opt = st.checkbox("Normalize Heatmap", value=True)
        cmap_opt = st.selectbox("Heatmap Colormap", ["viridis","mako","rocket","coolwarm"])
        fig_h, axh = plt.subplots(figsize=(5,4))
        data_h = visit_counts / visit_counts.max() if (norm_opt and visit_counts.max()>0) else visit_counts
        sns.heatmap(data_h, cmap=cmap_opt, ax=axh)
        axh.set_title("Visit Frequency Heatmap")
        st.pyplot(fig_h)
        plt.clf()

# Top 5 shortest successful
        success_eps = df[df['success']==1].nsmallest(5,'steps')[['episode','steps','reward']]
        st.write("🏆 Top 5 Fastest Successes:")
        st.table(success_eps)

# Episode replay slider 
        if len(df) > 0:
            sel_ep = st.slider("Replay Episode", 1, len(df), 1)
            selected_traj = df.loc[sel_ep-1,'trajectory']
            replay_env = GridEnv(size=grid_size, obstacle_prob=obstacle_prob, obs_window=OBS_WINDOW, max_steps=max_steps)
            replay_env.reset()
            replay_positions = []
# draw step by step
            for pos in selected_traj:
# set agent position and keep start/goal from the replayed env
                replay_env.agent_pos = pos
                replay_positions.append(pos)
                plot_grid(replay_env, replay_positions, figsize=(5,5))
                st.pyplot(plt.gcf())
                plt.clf()
                time.sleep(0.03)

# Auto-save results and provide downloads
        os.makedirs("results", exist_ok=True)
        csv_path = "results/batch_summary.csv"
        heatmap_path = "results/visit_heatmap.png"
        df.to_csv(csv_path, index=False)
# Save heatmap image
        fig_h2, axh2 = plt.subplots(figsize=(5,4))
        sns.heatmap(data_h, cmap=cmap_opt, ax=axh2)
        axh2.set_title("Visit Frequency Heatmap")
        fig_h2.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.clf()

        st.success("Results saved to results/")
        with open(csv_path, "rb") as f:
            st.download_button("Download batch_summary.csv", f.read(), file_name="batch_summary.csv", mime="text/csv")
        with open(heatmap_path, "rb") as f:
            st.download_button("Download visit_heatmap.png", f.read(), file_name="visit_heatmap.png", mime="image/png")

# Save current sim snapshot

st.markdown("---")
st.header("Utilities")
colx, coly = st.columns([1,1])
with colx:
    if st.button("Save Current Simulation Snapshot (PNG)"):
        if st.session_state.sim_state:
            env_snapshot = st.session_state.sim_state["env"]
            path = "snapshot.png"
            plot_grid(env_snapshot, st.session_state.sim_state["positions"], figsize=(5,5))
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.clf()
            with open(path, "rb") as f:
                st.download_button("Download PNG", f.read(), file_name="snapshot.png", mime="image/png")
        else:
            st.warning("No active simulation to snapshot.")

with coly:
    if st.button("Show App Tips"):
        st.info(
            "Tips:\n\n"
            "- Set Grid Size & Obstacle Prob in sidebar.\n"
            "- OBS_WINDOW is fixed to the trained model (7).\n"
            "- Use 'Next Step' for debugging, 'Auto Run' for demo.\n"
            "- Run Batch Evaluation to generate summary table and visit heatmap."
        )

# Inspirational Quotes 
st.markdown("---")
st.subheader("Words of Wisdom")
quotes = [
    ("When something is important enough, you do it even if the odds are not in your favor.", "Elon Musk"),
    ("The best way to predict the future is to invent it.", "Alan Kay"),
    ("Somewhere, something incredible is waiting to be known.", "Carl Sagan")
]
for q, a in quotes:
    st.markdown(f'<div class="quote">“{q}”<br><span class="author">— {a}</span></div>', unsafe_allow_html=True)

st.caption("Built with Streamlit • Seaborn • PyTorch — Open-source. Customize as you like.")
