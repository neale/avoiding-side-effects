import os
from scipy import interpolate
import numpy as np
from gym import spaces

from safelife.safelife_env import SafeLifeEnv
from safelife.safelife_game import CellTypes
from safelife import env_wrappers
 
from safelife.render_graphics import render_board, render_game
from safelife.helper_utils import recenter_view
from types import SimpleNamespace
 
 
class SafeLifeRGBEnv(SafeLifeEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self.view_shape[0]*14, self.view_shape[1]*14, 3),
            dtype=np.uint8)
        print ('spinning up teh rgb env')
    
    def get_obs(self, board=None, goals=None, agent_loc=None):
        if board is None:
            board = self.game.board
        if goals is None:
            goals = self.game.goals
        if agent_loc is None:
            agent_loc = self.game.agent_loc
 
        import matplotlib.pyplot as plt
        board = recenter_view(
            board, (15, 15), agent_loc[::-1], self.game.exit_locs)
        goals = recenter_view(
            goals, (15, 15), agent_loc[::-1], self.game.exit_locs)
        orientation = self.game.orientation
        
        state = render_board(board, goals, orientation)
        return state


def linear_schedule(t, y):
    """
    Piecewise linear function y(t)
    """
    return interpolate.UnivariateSpline(t, y, s=0, k=1, ext='const')


def safelife_env_factory(
        logdir, level_iterator, *,
        num_envs=1,
        min_performance=None,
        summary_writer=None,
        impact_penalty=None,
        testing=False):
    """
    Factory for creating SafeLifeEnv instances with useful wrappers.
    """
    if testing:
        video_name = "test-{level_title}-{step_num}"
        tag = "episodes/testing/"
        log_name = "testing.yaml"
        log_header = "# Testing episodes\n---\n"
    else:
        video_name = "training-episode-{episode_num}-{step_num}"
        tag = "episodes/training/"
        log_name = "training.yaml"
        log_header = "# Training episodes\n---\n"

    logdir = os.path.abspath(logdir)
    video_name = os.path.join(logdir, video_name)
    log_name = os.path.join(logdir, log_name)

    if os.path.exists(log_name):
        log_file = open(log_name, 'a')
    else:
        log_file = open(log_name, 'w')
        log_file.write(log_header)

    envs = []
    for _ in range(num_envs):
        env = SafeLifeEnv(
        #env = SafeLifeRGBEnv(
            level_iterator,
            view_shape=(25,25),
            # This is a minor optimization, but a few of the output channels
            # are redundant or unused for normal safelife training levels.
            output_channels=(
                CellTypes.alive_bit,
                CellTypes.agent_bit,
                CellTypes.pushable_bit,
                CellTypes.destructible_bit,
                CellTypes.frozen_bit,
                CellTypes.spawning_bit,
                CellTypes.exit_bit,
                CellTypes.color_bit + 0,  # red
                CellTypes.color_bit + 1,  # green
                CellTypes.color_bit + 5,  # blue goal
            ))
        other_data = {}

        if testing:
            env.global_counter = None  # don't increment num_steps
        else:
            env = env_wrappers.MovementBonusWrapper(env, as_penalty=True)
            env = env_wrappers.ExtraExitBonus(env)
        if impact_penalty is not None:
            env = env_wrappers.SimpleSideEffectPenalty(
                env, penalty_coef=impact_penalty)
            if not testing:
                other_data = {'impact_penalty': impact_penalty}
        if min_performance is not None:
            env = env_wrappers.MinPerformanceScheduler(
                env, min_performance=min_performance)
        env = env_wrappers.RecordingSafeLifeWrapper(
            env, video_name=video_name, summary_writer=summary_writer,
            log_file=log_file, other_episode_data=other_data, tag=tag,
            video_recording_freq=1 if testing else 50,
            exclude=('num_episodes', 'performance_cutoff') if testing else ())
        # Ensure the recording wrapper has access to the global counter,
        # even if it's disabled in the unwrapped environment.
        env.global_counter = SafeLifeEnv.global_counter
        
        env.global_counter.episodes_started=0
        env.global_counter.episodes_completed=0
        env.global_counter.num_steps=0

        envs.append(env)

    return envs
