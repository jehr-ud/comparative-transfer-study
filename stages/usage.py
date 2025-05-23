from pathlib import Path
from datetime import datetime

import time
import pygame
from PIL import Image
import numpy as np

from .utils import load_model
from environments import create_env


def run_agent_episode_and_record_gif(
    model_name,
    agent,
    env,
    gif_path="maze_run.gif",
    sleep_time=0.1
):
    obs = env.reset()
    done = False
    frames = []
    steps = 0
    total_reward = 0.0

    env.render()

    while not done:
        print(f"Using: {model_name}")
        if isinstance(obs, dict):
            action = agent.predict(obs)["agent_0"]
        else:
            action = agent.predict(obs)

        print("Action predicted:", action)

        obs, reward, done, info = env.step(action)
        print("info:", info)
        total_reward += reward
        steps += 1

        env.render()

        # screen recording
        frame_surface = pygame.display.get_surface()
        frame_array = pygame.surfarray.array3d(frame_surface)
        frame_image = Image.fromarray(np.transpose(frame_array, (1, 0, 2)))
        frames.append(frame_image)
        pygame.event.pump()
        time.sleep(sleep_time)

    print(f"🎥 Saving GIF with {len(frames)} frames in: {gif_path}")
    print(f"Go to {gif_path}.gif")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(sleep_time * 1000),
        loop=0
    )

    print(
        f"✅ Episode finish in {steps} steps, with total reward: {total_reward}"
    )


def show_agent(algoritmhs, difficulty):
    pygame.init()

    screen_width = 700
    screen_height = 700
    screen = pygame.display.set_mode((screen_width, screen_height), 0, 32)
    screen.fill((255, 255, 255))
    pygame.display.set_caption("Agent movements in Maze")

    for algoritmh in algoritmhs:
        model_name = algoritmh.get('name')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        env = create_env(
            difficulty,
            "human",
            f"results/monitor/usage_{model_name}_{timestamp}",
            shared_window=screen
        )

        if env:
            print(f"Env found: {env}")
        else:
            print("Env not found.")
            return

        path_models = str(Path("models"))
        model_path = f"{path_models}/{model_name}_{difficulty}"

        print(f"Loading {model_path}")
        agent = load_model(
                model_path,
                algoritmh.get('class'),
                env
        )

        run_agent_episode_and_record_gif(
            model_name,
            agent,
            env,
            gif_path=f"results/run_maze_{difficulty}_{model_name}.gif"
        )
