"""Minimal training loop placeholder using a policy-gradient style update.
This file is a starting point — replace with DQN/PPO/A2C as desired.
"""
import torch
import torch.optim as optim
from agent.model import AutoScaleAgent
import numpy as np
import argparse
import torch.distributions



class Trainer:
    def __init__(self, state_dim, action_dim, lr=1e-3):
        self.agent = AutoScaleAgent(state_dim, action_dim)
        self.optimizer = optim.Adam(self.agent.policy.parameters(), lr=lr)


    def compute_loss(self, logits, actions, advantages):
        logp = torch.log_softmax(logits, dim=-1)
        selected = logp.gather(1, actions.unsqueeze(1)).squeeze(1)
        return -(selected * advantages).mean()


    def train_step(self, batch_states, batch_actions, batch_advantages):
        self.agent.policy.train()
        logits = self.agent.policy(torch.tensor(batch_states, dtype=torch.float32))
        loss = self.compute_loss(logits, torch.tensor(batch_actions, dtype=torch.long), torch.tensor(batch_advantages, dtype=torch.float32))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


def _get_state(self):
    # normalize values to [0,1]
    max_rps = 200.0
    s = np.array([
        min(1.0, self.rps / max_rps),
        self.replicas / self.max_replicas,
        0.0 # placeholder for cpu util; computed after step
    ], dtype=np.float32)
    return s


def train(env, agent, epochs=100, gamma=0.99, lr=1e-3):
    optimizer = optim.Adam(agent.policy.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        log_probs = []
        rewards = []
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            logits = agent.policy(state_tensor.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            logp = dist.log_prob(action)

            next_state, reward, done, info = env.step(int(action.item()))

            log_probs.append(logp)
            rewards.append(reward)
            total_reward += reward
            state = next_state

        # compute discounted returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32)
        # normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for logp, R in zip(log_probs, returns):
            loss = loss - logp * R

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % 10 == 0: # simple metrics print
            last = env.history[-1] if env.history else {}
            print(f"Ep {ep:03d} | total_reward={total_reward:.2f} | replicas={last.get('replicas')} | latency={last.get('latency'):.1f}ms | rps={last.get('rps'):.1f}")

    return agent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['sim', 'real'], default='sim')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    if args.mode == 'sim':
        from env.simulated_env import SimulatedEnvironment
        env = SimulatedEnvironment()
        state_dim = 3
        action_dim = 3
        agent = AutoScaleAgent(state_dim, action_dim)
        agent = train(env, agent, epochs=args.epochs)
        agent.save('autoscale_agent.pt')
        print('Training finished. Model saved to autoscale_agent.pt')
    else:
        print('Real mode is not implemented in this scaffold. Use --mode sim to run the local training.')