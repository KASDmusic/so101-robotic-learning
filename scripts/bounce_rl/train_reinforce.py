import sys
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import imageio.v2 as imageio

ROOT = Path.cwd().resolve().parent.parent.parent
sys.path.append(str(ROOT / "src"))

from so101_robotic_learning.bounce_rl.env.bounce_env import BounceEnv


class PingPongImageEmbedding(nn.Module):
    def __init__(self, image_shape, embedding_dim=64):
        super().__init__()
        c, h, w = image_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        conv_output_dim = self._get_conv_output_dim(image_shape)
        self.fc = nn.Linear(conv_output_dim, embedding_dim)

    def _get_conv_output_dim(self, shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            conv_output = self.conv(dummy_input)
            return int(np.prod(conv_output.shape))

    def forward(self, image):
        x = self.conv(image)
        x = x.view(x.size(0), -1)
        embedding = self.fc(x)
        return embedding


class PingPongModel(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super().__init__()
        
        #self.image_embedding = PingPongImageEmbedding(image_shape=(3, 240, 320), embedding_dim=64)

        self.backbone = nn.Sequential(
            #nn.Linear(obs_shape[0] - 3 * 240 * 320 + 64, 128),
            nn.Linear(observation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))



    def forward(self, state):

        #image_emb = self.image_embedding(image)
        #x = torch.cat([state, image_emb], dim=-1)
        x = state

        features = self.backbone(x)
        mean = self.mean_head(features)
        std = torch.exp(self.log_std)
        return mean, std


def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    returns = torch.tensor(returns, dtype=torch.float32)

    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return returns


def save_video(frames, output_path, fps=30):
    if not frames:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)


def train():
    env = BounceEnv(
        xml_path=str(ROOT / "assets" / "mjcf" / "so101_new_calib copy.xml"),
        render_mode=None,
        
    )

    video_dir = ROOT / "videos"
    video_every = 10

    state_shape = env.observation_space["state"].shape
    print("State shape:", state_shape)
    image_shape = env.observation_space["image"].shape
    action_dim = env.action_space.shape[0]
    print("Action dim:", action_dim)

    input_dim = int(np.prod(state_shape) + np.prod(image_shape))

    model = PingPongModel(int(np.prod(state_shape)), action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_episodes = 1000
    gamma = 0.99
    max_steps_per_episode = 50

    for episode in range(num_episodes):

        t_episode = time.time()
        t_inferance = 0.0
        t_simulation = 0.0

        should_record = ((episode + 1) % video_every == 0)

        # On change le mode de rendu avant reset pour bien vider/recréer le buffer de frames
        env.render_mode = "rgb_array_list" if should_record else None

        obs, info = env.reset()

        log_probs = []
        rewards = []
        total_reward = 0.0

        for step in range(max_steps_per_episode):
            state = torch.tensor(obs["state"], dtype=torch.float32).unsqueeze(0)
            image = torch.tensor(obs["image"], dtype=torch.float32).unsqueeze(0) / 255.0

            states = torch.cat([state, image.reshape(image.shape[0], -1)], dim=-1)
            
            t0_inferance = time.time()
            mean, std = model(state)
            t_inferance += time.time() - t0_inferance

            dist = Normal(mean, std)
            action = dist.sample()
            print(action)
            log_prob = dist.log_prob(action).sum(dim=-1)

            action_np = torch.clamp(action, -1.0, 1.0).squeeze(0).detach().cpu().numpy()

            t0_simulation = time.time()
            obs, reward, terminated, truncated, info = env.step(action_np)
            t_simulation += time.time() - t0_simulation
            
            done = terminated or truncated

            log_probs.append(log_prob.squeeze(0))
            rewards.append(float(reward))
            total_reward += float(reward)

            if done:
                break

        returns = compute_returns(rewards, gamma=gamma)

        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)

        loss = torch.stack(policy_loss).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if should_record:
            frames = env.render()
            video_path = video_dir / f"episode_{episode + 1:04d}.mp4"
            save_video(frames, video_path, fps=int(env.camera_fps))
            print(f"Vidéo enregistrée : {video_path}")

        t_episode = time.time() - t_episode

        print(
            f"Episode {episode + 1:04d} | "
            f"steps={len(rewards):03d} | "
            f"total_reward={total_reward:.3f} | "
            f"loss={loss.item():.3f} | "
            f"inference_time={t_inferance:.3f}s | "
            f"simulation_time={t_simulation:.3f}s | "
            f"episode_time={t_episode:.3f}s"
        )

    env.close()