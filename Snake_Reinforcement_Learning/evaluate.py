import time
from stable_baselines3 import PPO
from snek_env import SnekEnv

def evaluate(model_path, episodes=5):
    env = SnekEnv(render_mode=True)
    model = PPO.load(model_path)

    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward

            env.render()
            time.sleep(0.05)  # Ajusta velocidad de render

        print(f"Episodio {ep + 1} - Recompensa total: {total_reward}")

    env.close()

if __name__ == "__main__":
    evaluate("ppo_snake_final.zip")  # O ruta al modelo guardado
