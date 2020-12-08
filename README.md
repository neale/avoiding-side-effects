# Code accompying the paper _Avoiding Side Effects in Complex Environments_ 

This repository contains the main results and code for reproducing the experiments performed in the paper. 

We also include pretrained models for each tested method on each Safelife task.

The paper can be found [on arxiv](https://arxiv.org/abs/2006.06547)


## Abstract

Reward function specification can be difficult. Rewarding the agent for making a widget may be easy, but penalizing the multitude of possible negative side effects is hard. In toy environments,  Attainable Utility Preservation (AUP) avoided side effects by penalizing shifts in the ability to achieve randomly generated goals. We scale this approach to large, randomly generated environments based on Conway's Game of Life. By preserving optimal value for a single randomly generated reward function, AUP incurs modest overhead while leading the agent to complete the specified task and avoid many side effects.

SafeLife is a novel environment to test the safety of reinforcement learning agents. The long term goal of this project is to develop training environments and benchmarks for numerous technical reinforcement learning safety problems, with the following attributes:




## Usage

Install the Safelife environment by following the instructions on [their repository](https://github.com/PartnershipOnAI/safelife) 

Alternatively, here are some basic instructions for a local install

    pip3 install -r requirements.txt
    python3 setup.py build_ext --inplace

Note that we use version 1.0 of Safelife. Some large changes that we have not thoroughly tested, were implemented in the current master branch of Safelife


## Training an agent

The `train` script is an easy way to get agents up and running using the default proximal policy optimization implementation. Just run

    ./train --algo aup

to start training. Saved files including checkpoints, logging file, and intermediate episode videos are stored in `data/aup/<task>`. 

## Loading a Saved Model

We include saved models for AUP and the PPO baseline, for each SafeLife task. 

### Continuing Training with a Model Checkpoint

### Generating Agent Videos with a Model Checkpoint 

## Results on SafeLife Tasks

We trained agents on four different Safelife tasks. Two of our tasks involve placing cells on goal tiles, with an initially static board. In this scenario, the board is initialized with many (`append_still`), or fewer green cells (`append_still-easy`). The third task considers the same goal, but the board initializes with dynamic yellow cells that spawn more cells (`append_spawn`). In the final task, the agent is tasked with removing red cell patterns from the initially static board (`prune-still`). We show the main results (reward and side-effects) below, for all considered methods, on each task. 

GIF files for each task can be found in the [GIFs](https://github.com/neale/avoiding-side-effects/tree/master/gifs) directory.  

### Append_Still-Easy Results 
<p align="center">
<img alt="main results" src="https://github.com/neale/avoiding-side-effects/blob/master/figures/side_effect_append_still-easy_plot.png" width=275/>
<img alt="main results" src="https://github.com/neale/avoiding-side-effects/blob/master/figures/reward_append_still-easy_plot.png" width=275/>
<img alt="main results" src="https://github.com/neale/avoiding-side-effects/blob/master/figures/length_append_still-easy_plot.png" width=275/>
</p>


### Append_Still Results 
<p align="center">
<img alt="main results" src="https://github.com/neale/avoiding-side-effects/blob/master/figures/side_effect_append_still_plot.png" width=275/>
<img alt="main results" src="https://github.com/neale/avoiding-side-effects/blob/master/figures/reward_append_still_plot.png" width=275/>
<img alt="main results" src="https://github.com/neale/avoiding-side-effects/blob/master/figures/length_append_still_plot.png" width=275/>
</p>


### Append_Spawn Results 
<p align="center">
<img alt="main results" src="https://github.com/neale/avoiding-side-effects/blob/master/figures/side_effect_append_spawn_plot.png" width=275/>
<img alt="main results" src="https://github.com/neale/avoiding-side-effects/blob/master/figures/reward_append_spawn_plot.png" width=275/>
<img alt="main results" src="https://github.com/neale/avoiding-side-effects/blob/master/figures/length_append_spawn_plot.png" width=275/>
</p>


### Prune_Still-Easy Results 
<p align="center">
<img alt="main results" src="https://github.com/neale/avoiding-side-effects/blob/master/figures/side_effect_prune_still-easy_plot.png" width=275/>
<img alt="main results" src="https://github.com/neale/avoiding-side-effects/blob/master/figures/reward_prune_still-easy_plot.png" width=275/>
<img alt="main results" src="https://github.com/neale/avoiding-side-effects/blob/master/figures/length_prune_still-easy_plot.png" width=275/>
</p>

