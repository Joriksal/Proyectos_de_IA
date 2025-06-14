import os
import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from snek_env import SnekEnv

MODEL_PATH = "models/PPO_Snake/ppo_snake_final.zip"
N_EPISODES = 50

def evaluate_agent(model_path, episodes=50, render=False):
    env = SnekEnv(render_mode=render)
    model = PPO.load(model_path)

    rewards = []
    steps_per_episode = []

    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            steps += 1

            if render:
                env.render()

        rewards.append(ep_reward)
        steps_per_episode.append(steps)
        print(f"Episodio {ep+1}: Recompensa = {ep_reward:.2f}, Pasos = {steps}")

    env.close()
    return rewards, steps_per_episode

def plot_metrics(rewards, steps):
    os.makedirs("results", exist_ok=True)

    # Recompensas
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, marker='o')
    plt.title("Recompensa por episodio")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")
    plt.grid(True)
    plt.savefig("results/recompensas.png")
    plt.close()

    # Pasos por episodio
    plt.figure(figsize=(10, 5))
    plt.plot(steps, marker='x', color='orange')
    plt.title("Pasos por episodio")
    plt.xlabel("Episodio")
    plt.ylabel("Pasos")
    plt.grid(True)
    plt.savefig("results/pasos_por_episodio.png")
    plt.close()

    # Estadísticas
    print("\n--- Estadísticas ---")
    print(f"Recompensa promedio: {np.mean(rewards):.2f}")
    print(f"Recompensa máxima: {np.max(rewards):.2f}")
    print(f"Pasos promedio: {np.mean(steps):.2f}")

if __name__ == "__main__":
    rewards, steps = evaluate_agent(MODEL_PATH, episodes=N_EPISODES, render=False)
    plot_metrics(rewards, steps)
