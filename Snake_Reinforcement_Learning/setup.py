from setuptools import setup, find_packages

def parse_requirements(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    reqs = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
    return reqs

setup(
    name = "snake_rl",
    version = "0.1.0",
    description = "Implementación y entrenamiento de un agente de aprendizaje por refuerzo PPO en un entorno personalizado tipo Snake, utilizando Stable Baselines 3.",
    author = "Salas Castañon Jose Ricardo",
    packages = find_packages(),
    install_requires = parse_requirements("requirements.txt"),
    python_requires = ">=3.7",
)