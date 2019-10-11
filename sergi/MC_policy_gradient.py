import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm

def select_action(model, state):
    # Samples an action according to the probability distribution induced by the model
    # Also returns the log_probability
    log_p = model.forward(torch.Tensor(state))
    action = torch.multinomial(torch.exp(log_p), 1).item()
    return action, log_p[action]


def run_episode(env, model):
    episode = []
    S = env.reset()
    done = False
    while not done:
        A, log_p = select_action(model, S)
        S_next, R_step, done, P = env.step(A)
        episode.append((R_step, log_p))
        S = S_next
    return episode


def compute_reinforce_loss(episode, discount_factor):
    # Compute the reinforce loss
    # Make sure that your function runs in LINEAR TIME
    # Don't forget to normalize your RETURNS (not rewards)
    # Note that the rewards/returns should be maximized 
    # while the loss should be minimized so you need a - somewhere
    G = 0
    G_list = []    
    rewards_reversed, log_ps = zip(*reversed(episode))
    
    for R_step in rewards_reversed:
        G = R_step + discount_factor * G
        G_list.append(G)

    Gs = torch.Tensor(G_list)
    log_ps = torch.stack(log_ps)
    G_normalized = (Gs - Gs.mean()) / Gs.std()
    loss = -(log_ps * G_normalized).sum()

    return loss


def run_episodes_policy_gradient(model, env, num_episodes, discount_factor, learn_rate):
    
    optimizer = optim.Adam(model.parameters(), learn_rate)
    
    episode_durations = []
    for i in range(num_episodes):
        optimizer.zero_grad()
        episode = run_episode(env, model)
        loss = compute_reinforce_loss(episode, discount_factor)
        
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print("{2} Episode {0} finished after {1} steps"
                  .format(i, len(episode), '\033[92m' if len(episode) >= 195 else '\033[99m'))
            
        episode_durations.append(len(episode))
        
    return episode_durations
