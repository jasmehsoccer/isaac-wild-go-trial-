import os
import sys
import time

from isaacgym import gymapi
from src.configs.defaults import asset_options as asset_options_config
from src.envs.terrains.utils import load_cement_road_asset, load_stone_asset
from src.envs.terrains.utils import random_quaternion, add_uneven_terrains

import torch
import numpy as np
from typing import Any, List
import ml_collections


class WildTerrainEnv:
    def __init__(
            self,
            sim: Any,
            gym: Any,
            viewer: Any,
            env_handle: Any,
    ):
        """Initialize the wild terrain env class."""

        self._sim = sim
        self._gym = gym
        self._viewer = viewer
        self._enable_viewer_sync = True
        # self._sim_config = sim_config
        # self._device = self._sim_config.sim_device
        self._actors = []
        self._env = env_handle

        # if "cuda" in self._device:
        #     torch._C._jit_set_profiling_mode(False)
        #     torch._C._jit_set_profiling_executor(False)

        self._load_all_assets()
        # 初次渲染
        # gym.simulate(self._sim)
        # gym.fetch_results(sim, True)
        # gym.step_graphics(sim)
        # gym.draw_viewer(self._viewer, sim, True)
        #
        # # 持续渲染
        # while not gym.query_viewer_has_closed(self._viewer):
        #     gym.simulate(sim)
        #     gym.fetch_results(sim, True)
        #     gym.step_graphics(sim)
        #     gym.draw_viewer(self._viewer, sim, True)
        #
        # # 清理资源
        # gym.destroy_viewer(self._viewer)
        # gym.destroy_sim(sim)
        # time.sleep(123)
        # self._gym.prepare_sim(self._sim)

        # self._post_physics_step()

    def _init_buffers(self):
        pass

    def _load_all_assets(self):

        # Add outdoor scene
        self.load_outdoor_asset(env=self._env, reverse=False)

        # Add uneven terrains
        add_uneven_terrains(gym=self._gym, sim=self._sim, reverse=False)

    def load_outdoor_asset(self, env, scene_offset_x=40, reverse=False):
        if reverse:
            offset_x = -28 + scene_offset_x
        else:
            offset_x = 0 + scene_offset_x
        offset_y = 4

        # Directional Arrow
        # load_arrow_asset(gym=gym, sim=sim, i=0, pos=(3 + offset_x, -2 + offset_y, 0.), rot=(1, 0, 0, 0),
        #                  apply_texture=False, scale=1.2)
        # load_arrow_asset(gym=gym, sim=sim, i=1, pos=(3 + offset_x, -6 + offset_y, 0.), rot=(1, 0, 0, 0),
        #                  apply_texture=True, scale=1.2)

        # Cement Road
        if reverse:
            actor = load_cement_road_asset(self._gym, self._sim, env=env,
                                           pos=(-5 + scene_offset_x, -4 + offset_y, 0.001),
                                           apply_texture=True, scale=1.0)
        else:
            actor = load_cement_road_asset(self._gym, self._sim, env=env,
                                           pos=(4 + scene_offset_x + 1, -4 + offset_y, 0.001),
                                           apply_texture=True, scale=1.0)
        self._actors.append(actor)

        # Mountain Rocks
        # self.load_mountain_rocks(env=env, offset_x=offset_x, offset_y=offset_y)

        # Random stones
        # self.load_random_snowstones_in_a_region(env=env, stone_nums=450, reverse=reverse)
        # self.load_random_cobblestones_in_a_region(env=env, stone_nums=150, reverse=reverse)

    def load_mountain_rocks(self, env, offset_x, offset_y):
        # Big Snow Rocks
        actor1 = load_stone_asset(self._gym, self._sim, env=env, name="Mountain Rock1",
                                  pos=(12 + offset_x, -0.5 + offset_y, 0.1), rot=(0, 0, 1, 0),
                                  apply_texture=False, scale=0.25)
        actor2 = load_stone_asset(self._gym, self._sim, env=env, name="Mountain Rock2",
                                  pos=(14 + offset_x, offset_y, 0.1), rot=(0.3, 0.2, 0.1, 1),
                                  apply_texture=False, scale=0.3)
        actor3 = load_stone_asset(self._gym, self._sim, env=env, name="Mountain Rock3",
                                  pos=(17 + offset_x, -1 + offset_y, 0.1), rot=(0.2, 0.1, 0.8, 0),
                                  apply_texture=False, scale=0.35)
        actor4 = load_stone_asset(self._gym, self._sim, env=env, name="Mountain Rock4",
                                  pos=(10 + offset_x, offset_y, 0.1), rot=(0., 0., 1., 0.),
                                  apply_texture=False, scale=0.4)
        actor5 = load_stone_asset(self._gym, self._sim, env=env, name="Mountain Rock5",
                                  pos=(18 + offset_x, -8 + offset_y, 0.1), rot=(0, 0, 0, 1),
                                  apply_texture=False, scale=0.4)
        actor6 = load_stone_asset(self._gym, self._sim, env=env, name="Mountain Rock6",
                                  pos=(10 + offset_x, -7 + offset_y, 0.3), rot=(0.2, 0, 0.8, 0),
                                  apply_texture=False, scale=0.3)
        actor7 = load_stone_asset(self._gym, self._sim, env=env, name="Mountain Rock7",
                                  pos=(13 + offset_x, -8.5 + offset_y, 0.1), rot=(0.2, 0, 0.8, 0),
                                  apply_texture=False, scale=0.55)
        actor8 = load_stone_asset(self._gym, self._sim, env=env, name="Mountain Rock8",
                                  pos=(10 + offset_x, -3 + offset_y, 0.1), rot=(0.2, 0, 0.8, 0),
                                  apply_texture=False, scale=0.25)

        self._actors.append(actor1)
        self._actors.append(actor2)
        self._actors.append(actor3)
        self._actors.append(actor4)
        self._actors.append(actor5)
        self._actors.append(actor6)
        self._actors.append(actor7)
        self._actors.append(actor8)

    def load_random_snowstones_in_a_region(self, env, scene_offset_x=40, width=4.0, length=5.0, stone_nums=10,
                                           reverse=False):
        if reverse:
            offset_x = -13 + scene_offset_x
        else:
            offset_x = 8.1 + scene_offset_x
        offset_y = -1.8
        np.random.seed(25)
        for i in range(stone_nums):
            random_width = np.random.uniform(low=0, high=width)
            random_length = np.random.uniform(low=0, high=length)
            random_quat = random_quaternion()
            scale = np.random.uniform(low=0.1, high=0.18)

            x = offset_x + random_length
            y = offset_y + random_width

            stone_actor = load_stone_asset(self._gym, self._sim, env=env, name=f"snow_stones{i}", pos=(x, y, 0.03),
                                           fix_base_link=False, rot=tuple(random_quat), apply_texture=False,
                                           scale=scale)
            self._actors.append(stone_actor)

    def load_random_cobblestones_in_a_region(self, env, scene_offset_x=40, width=2.0, length=2.0, stone_nums=10,
                                             reverse=False):
        if reverse:
            offset_x = -8.1 + scene_offset_x
        else:
            offset_x = 6 + scene_offset_x
        offset_y = -0.985
        np.random.seed(12)
        for i in range(stone_nums):
            random_width = np.random.uniform(low=0, high=width)
            random_length = np.random.uniform(low=0, high=length)
            random_quat = random_quaternion()
            scale = np.random.uniform(low=0.06, high=0.08)

            x = offset_x + random_length
            y = offset_y + random_width

            stone_actor = load_stone_asset(self._gym, self._sim, env=env, pos=(x, y, 0.007), name=f"cobblestones{i}",
                                           fix_base_link=True, rot=tuple(random_quat), apply_texture=True, scale=scale)
            self._actors.append(stone_actor)

    def init_buffers(self):
        self._init_buffers()
