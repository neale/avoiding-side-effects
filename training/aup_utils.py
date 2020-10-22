import torch
import numpy as np



def get_rand_reward(state, encoder, random_reward_fns):
    state_z = encoder(state)
    rewards = []
    for reward_fn in random_reward_fns:
        r = torch.dot(reward_fn, state_z)
        rewards.append(r)
    rewards = torch.stack(rewards)
    return rewards

def 
