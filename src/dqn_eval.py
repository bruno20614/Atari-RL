import random
from typing import Callable

import gymnasium as gym
import numpy as np
import torch


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episode: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = True
):
    # Cria ambiente vetorizado (só 1 instância neste caso)
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    
    # Carrega modelo
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episode:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = model(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, _, _, _, infos = envs.step(actions)
        
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns.append(info["episode"]["r"])
        
        obs = next_obs

    return episodic_returns


if __name__ == "__main__":
    from dqn_atari import QNetwork, make_env  # seus módulos locais

    # 🔹 Carrega modelo salvo localmente
    model_path = "dqn_model.pth"  # caminho local do modelo salvo

    episodic_returns = evaluate(
        model_path=model_path,
        make_env=make_env,
        env_id="CartPole-v1",
        eval_episode=5,
        run_name="eval",
        Model=QNetwork,
        device="cpu",
        capture_video=False
    )

    print("Episodic returns:", episodic_returns)
