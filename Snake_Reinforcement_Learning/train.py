import gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from snek_env import SnekEnv  # Asegúrate que snek_env.py está en el mismo folder

# Crear entorno env con render_mode=False para entrenamiento
def make_env():
    return SnekEnv(render_mode=False)

env = DummyVecEnv([make_env])

# Definir el modelo PPO con CNN policy para imagen RGB
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_snake_tensorboard/",
    device="cuda" if torch.cuda.is_available() else "cpu",
    learning_rate=2.5e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
)

# Checkpoint para guardar cada 10000 pasos
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="./models/",
                                         name_prefix="ppo_snake")

# Entrenar el modelo por 1 millón de pasos (puedes ajustar)
model.learn(total_timesteps=1_000_000, callback=checkpoint_callback)

# Guardar modelo final
model.save("ppo_snake_final")

env.close()
