"""Policy outputs desired CoM speed for Go2 to track the desired speed."""

import time
import logging
from typing import Sequence

from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import to_torch
import ml_collections
import numpy as np
import torch

from src.configs.defaults import sim_config
from src.envs.go2_rewards import Go2Rewards
from src.envs.robots.modules.controller import raibert_swing_leg_controller, qp_torque_optimizer
from src.envs.robots.modules.controller.adaptive_gait_controller import AdaptiveGaitController
from src.envs.robots.modules.gait_generator import phase_gait_generator
from src.envs.robots import go2_robot, go2
from src.envs.robots.modules.planner.path_planner import PathPlanner
from src.envs.robots.motors import MotorControlMode, concatenate_motor_actions
from src.envs.terrains.wild_terrain_env import WildTerrainEnv
from src.ha_teacher.ha_teacher import HATeacher
from src.coordinator.coordinator import Coordinator
from omegaconf import DictConfig
import matplotlib.pyplot as plt

from src.utils.utils import ActionMode, RobotPusher


# @torch.jit.script
def torch_rand_float(lower, upper, shape: Sequence[int], device: str):
    return (upper - lower) * torch.rand(*shape, device=device) + lower


@torch.jit.script
def gravity_frame_to_world_frame(robot_yaw, gravity_frame_vec):
    cos_yaw = torch.cos(robot_yaw)
    sin_yaw = torch.sin(robot_yaw)
    world_frame_vec = torch.clone(gravity_frame_vec)
    world_frame_vec[:, 0] = (cos_yaw * gravity_frame_vec[:, 0] -
                             sin_yaw * gravity_frame_vec[:, 1])
    world_frame_vec[:, 1] = (sin_yaw * gravity_frame_vec[:, 0] +
                             cos_yaw * gravity_frame_vec[:, 1])
    return world_frame_vec


@torch.jit.script
def world_frame_to_gravity_frame(robot_yaw, world_frame_vec):
    cos_yaw = torch.cos(robot_yaw)
    sin_yaw = torch.sin(robot_yaw)
    gravity_frame_vec = torch.clone(world_frame_vec)
    gravity_frame_vec[:, 0] = (cos_yaw * world_frame_vec[:, 0] +
                               sin_yaw * world_frame_vec[:, 1])
    gravity_frame_vec[:, 1] = (sin_yaw * world_frame_vec[:, 0] -
                               cos_yaw * world_frame_vec[:, 1])
    return gravity_frame_vec


def create_sim(sim_conf):
    gym = gymapi.acquire_gym()
    _, sim_device_id = gymutil.parse_device_str(sim_conf.sim_device)
    if sim_conf.show_gui:
        graphics_device_id = sim_device_id
    else:
        graphics_device_id = -1

    sim = gym.create_sim(sim_device_id, graphics_device_id, sim_conf.physics_engine, sim_conf.sim_params)

    if sim_conf.show_gui:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "QUIT")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_V, "toggle_viewer_sync")
    else:
        viewer = None

    return gym, sim, viewer


class Go2WildExploreEnv:

    def __init__(self,
                 num_envs: int,
                 config: ml_collections.ConfigDict(),
                 device: str = "cuda",
                 show_gui: bool = False,
                 use_real_robot: bool = False):

        self._num_envs = num_envs
        self._device = device
        self._show_gui = show_gui
        self._config = config
        self._gamma = config.gamma
        self._terrain_region = to_torch(config.terrain_region, device=device)
        self._init_map_flag = True

        self._use_real_robot = use_real_robot

        with self._config.unlocked():
            if self._config.get('observation_noise', None) is not None:
                self._config.observation_noise = to_torch(
                    self._config.observation_noise, device=device)
            if self._config.get('action_noise', None) is not None:
                self._config.action_noise = to_torch(
                    self._config.action_noise, device=device)

        # Create simulation
        self._sim_conf = sim_config.get_config(use_gpu=True, show_gui=show_gui)
        self._gym, self._sim, self._viewer = create_sim(self._sim_conf)

        # Create terrain
        self._create_terrain()

        # Create robot
        init_positions = self._compute_init_positions()
        self._robot = go2.Go2(
            num_envs=num_envs,
            init_positions=init_positions,
            sim=self._sim,
            viewer=self._viewer,
            sim_config=self._sim_conf,
            world_env=WildTerrainEnv,
            motor_control_mode=MotorControlMode.HYBRID
        )

        # Coordinator
        self.coordinator = Coordinator(num_envs=self._num_envs, device=self._device)

        # HA-Teacher
        self.ha_teacher = HATeacher(num_envs=self._num_envs, teacher_cfg=config.ha_teacher, device=self._device)

        # Adaptive Gait Controller
        self._adaptive_gait_controller = AdaptiveGaitController(
            num_envs=num_envs,
            device=device
        )

        # Gait scheduler
        self._gait_generator = phase_gait_generator.PhaseGaitGenerator(self._robot, self._config.gait)

        # Reference state
        self.desired_vx = config.desired_vx
        self.desired_com_height = config.desired_pz
        self.desired_wz = config.desired_wz
        self.clip_wz = config.clip_wz

        # Swing leg controller
        self._swing_leg_controller = raibert_swing_leg_controller.RaibertSwingLegController(
            self._robot,
            self._gait_generator,
            desired_base_height=self.desired_com_height,
            foot_height=self._config.get('swing_foot_height', 0.12),
            foot_landing_clearance=self._config.get('swing_foot_landing_clearance', 0.02)
        )

        # Stance controller
        self._torque_optimizer = qp_torque_optimizer.QPTorqueOptimizer(
            self._robot,
            base_position_kp=self._config.get('base_position_kp', np.array([0., 0., 50.])),
            base_position_kd=self._config.get('base_position_kd', np.array([10., 10., 10.])),
            base_orientation_kp=self._config.get('base_orientation_kp', np.array([50., 50., 0.])),
            base_orientation_kd=self._config.get('base_orientation_kd', np.array([10., 10., 10.])),
            acc_lb=self._config.get('action_lb', np.array([-10, -10, -10, -20, -20, -20])),
            acc_ub=self._config.get('action_ub', np.array([10, 10, 10, 20, 20, 20])),
            weight_ddq=self._config.get('qp_weight_ddq', np.diag([20.0, 20.0, 5.0, 1.0, 1.0, .2])),
            foot_friction_coef=self._config.get('qp_foot_friction_coef', 0.7),
            clip_grf=self._config.get('clip_grf_in_sim') or self._use_real_robot,
            use_full_qp=self._config.get('use_full_qp', False)
        )

        # Quadruped Robot Pusher
        self._pusher = RobotPusher(
            robot=self._robot,
            sim=self._sim,
            viewer=self._viewer,
            num_envs=self._num_envs,
            device=self._device
        )
        self._push_flag = False

        # Set reference trajectory
        self._torque_optimizer.set_controller_reference(desired_height=self.desired_com_height,
                                                        desired_lin_vel=[self.desired_vx, 0, 0],
                                                        desired_rpy=[0., 0., 0.],
                                                        desired_ang_vel=[0., 0., self.desired_wz])
        # Path Planner
        self._planner = PathPlanner(robot=self._robot,
                                    sim=self._sim,
                                    viewer=self._viewer,
                                    num_envs=self._num_envs,
                                    device=self._device)
        self.sub_goal_reach_flag = False
        self.sub_goal_reach_time = []

        # Episodic statistics
        self._go2_reward = Go2Rewards(self, reward_cfg=config.reward)
        self._step_count = torch.zeros(self._num_envs, device=self._device)
        self._episode_length = self._config.episode_length_s / self._config.env_dt
        self._construct_observation_and_action_space()

        self._privileged_obs_buf = None
        self._obs_buf = torch.zeros((self._num_envs, config.obs_dim), device=self._device)
        self._last_obs_buf = torch.zeros((self._num_envs, config.obs_dim), device=self._device)
        self._last_action = torch.zeros((self._num_envs, config.act_dim), device=self._device)
        self._last_terminal_action = torch.zeros((self._num_envs, config.act_dim), device=self._device)
        self._goal_position = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float)
        self._cycle_count = torch.zeros(self._num_envs, device=self._device)
        self._teacher_activation_times = torch.zeros(self._num_envs, device=self._device)
        self._student_activation_times = torch.zeros(self._num_envs, device=self._device)

        # Initialize goal position
        self._goal_position[:, 0] = 10.0
        self._goal_position[:, 1] = 0.0

        # Initialize episode statistics
        self._episode_length_buf = torch.zeros(num_envs, device=device, dtype=torch.long)
        self._episode_count = torch.zeros(num_envs, device=device, dtype=torch.long)
        self._episode_rewards = torch.zeros(num_envs, device=device)
        self._episode_successes = torch.zeros(num_envs, device=device, dtype=torch.bool)

        # Initialize step count
        self._step_count = torch.zeros(num_envs, device=device, dtype=torch.long)
        self._cycle_count = torch.zeros(num_envs, device=device, dtype=torch.long)

        # Initialize action bounds
        self._action_lb = to_torch([-1.0] * 6, device=device)
        self._action_ub = to_torch([1.0] * 6, device=device)

        # Initialize observation and action spaces
        self._construct_observation_and_action_space()

        # Initialize activation times
        self._student_activation_times = torch.zeros(num_envs, device=device, dtype=torch.long)
        self._teacher_activation_times = torch.zeros(num_envs, device=device, dtype=torch.long)

        # Initialize last actions
        self._last_terminal_action = torch.zeros((num_envs, 6), device=device)

        # Initialize extras
        self._extras = {}

        # Initialize observation buffers
        self._obs_buf = torch.zeros((num_envs, self._config.obs_dim), device=device)
        self._privileged_obs_buf = torch.zeros((num_envs, self._config.obs_dim), device=device)
        self._last_obs_buf = torch.zeros((num_envs, self._config.obs_dim), device=device)
        self._last_action = torch.zeros((num_envs, self._config.act_dim), device=device)

        # Initialize desired velocities
        self.desired_vx = 0.5
        self.desired_wz = 0.0
        self.desired_com_height = 0.3

        # Initialize tracking error
        self._tracking_error = torch.zeros((num_envs, 6), device=device)

        # Initialize acceleration buffers
        self._desired_acc = torch.zeros((num_envs, 6), device=device)
        self._solved_acc = torch.zeros((num_envs, 6), device=device)
        self._qp_cost = torch.zeros((num_envs,), device=device)
        self._num_clips = torch.zeros((num_envs,), device=device)

        # Initialize terminal stance acceleration
        self._terminal_stance_ddq = torch.zeros((num_envs, 6), device=device)

        # Initialize logs
        self._logs = []

    def _create_terrain(self):
        """Creates terrains."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = .2
        plane_params.dynamic_friction = .2
        plane_params.restitution = 0.
        self._gym.add_ground(self._sim, plane_params)
        self._terrain = None

    def _compute_init_positions(self):
        init_positions = torch.zeros((self._num_envs, 3), device=self._device)
        num_cols = int(np.sqrt(self._num_envs))
        distance = 1.
        for idx in range(self._num_envs):
            init_positions[idx, 0] = idx // num_cols * distance
            init_positions[idx, 1] = idx % num_cols * distance
            init_positions[idx, 2] = 0.3
        return to_torch(init_positions, device=self._device)

    def _construct_observation_and_action_space(self):
        """Construct observation and action spaces with default values if not in config."""
        # Get observation dimensions from config or use defaults
        obs_dim = self._config.get('obs_dim', 100)  # Default observation dimension
        act_dim = self._config.get('act_dim', 6)    # Default action dimension (6 for Go2)
        
        # Create default observation bounds if not in config
        if hasattr(self._config, 'observation_lb') and hasattr(self._config, 'observation_ub'):
            self._observation_lb = to_torch(self._config.observation_lb, device=self._device)
            self._observation_ub = to_torch(self._config.observation_ub, device=self._device)
        else:
            # Default observation bounds: [-10, 10] for all dimensions
            self._observation_lb = to_torch([-10.0] * obs_dim, device=self._device)
            self._observation_ub = to_torch([10.0] * obs_dim, device=self._device)
        
        # Create default action bounds if not in config
        if hasattr(self._config, 'action_lb') and hasattr(self._config, 'action_ub'):
            self._action_lb = to_torch(self._config.action_lb, device=self._device)
            self._action_ub = to_torch(self._config.action_ub, device=self._device)
        else:
            # Default action bounds: [-1, 1] for all dimensions
            self._action_lb = to_torch([-1.0] * act_dim, device=self._device)
            self._action_ub = to_torch([1.0] * act_dim, device=self._device)

    def reset(self) -> torch.Tensor:
        return self.reset_idx(torch.arange(self._num_envs, device=self._device))

    def reset_idx(self, env_ids) -> torch.Tensor:
        # Aggregate rewards
        self._extras["time_outs"] = self._episode_terminated()
        if env_ids.shape[0] > 0:
            self._extras["episode"] = {}

            # Reset all rewards statistics
            for reward_name in self._go2_reward.episode_sums.keys():
                self._go2_reward.episode_sums[reward_name][env_ids] = 0

            self._step_count[env_ids] = 0
            self._cycle_count[env_ids] = 0

            self._robot.reset_idx(env_ids)
            self._swing_leg_controller.reset_idx(env_ids)
            self._gait_generator.reset_idx(env_ids)
            self._planner.reset()
            self._goal_position[env_ids, :2] = to_torch(self._planner.goal, dtype=torch.float32, device=self._device)

            self._obs_buf = self._get_observations()
            self._privileged_obs_buf = self._get_privileged_observations()
            self._last_obs_buf = torch.zeros((self._num_envs, self._config.obs_dim), device=self._device)
            self._last_action = torch.zeros((self._num_envs, self._config.act_dim), device=self._device)
            self._last_terminal_action = torch.zeros((self._num_envs, self._config.act_dim), device=self._device)

        return self._obs_buf, self._privileged_obs_buf

    def step(self, drl_action: torch.Tensor):
        drl_action = torch.clip(drl_action, self._action_lb, self._action_ub)
        nominal_actions = torch.zeros(self._num_envs, 6, device=self._device)
        sum_reward = torch.zeros(self._num_envs, device=self._device)
        dones = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)

        logs = []
        start = time.time()

        # Update gait generator and swing leg controller
        for _ in range(3):
            self._gait_generator.update()
            self._swing_leg_controller.update()
            desired_foot_positions = self._swing_leg_controller.desired_foot_positions
            self._torque_optimizer.get_action(self._gait_generator.desired_contact_state, self._swing_leg_controller.desired_foot_positions, torch.zeros_like(self._gait_generator.desired_contact_state[:, :6]), desired_contact_state, desired_foot_positions, torch.zeros_like(desired_contact_state[:, :6]), 
                self._gait_generator.desired_contact_state, self._swing_leg_controller.desired_foot_positions)

        # Compute model-based acceleration
        self._desired_acc, self._solved_acc, self._qp_cost, self._num_clips = self._torque_optimizer.compute_model_acc(
            foot_contact_state=self._gait_generator.desired_contact_state,
            desired_foot_position=self._swing_leg_controller.desired_foot_positions
        )

        # HP-Student action (residual form)
        raw_weights = to_torch([1.5, 0.5, 0.5, 0.1, 0.1, 0.5], device=self._device)
        weights = self._gamma
        drl_action_res = drl_action * weights
        phy_action_res = self._solved_acc * (1 - weights)
        hp_action = drl_action_res + phy_action_res

        # Update adaptive gait controller with tactile sensor data
        terrain_types = self._robot.tactile_sensor.detected_terrain_types
        terrain_confidence = self._robot.tactile_sensor.terrain_confidence
        friction_coefficients = self._robot.tactile_sensor.estimated_friction
        
        # Update gait parameters based on terrain
        adapted_gait_params = self._adaptive_gait_controller.update_gait_parameters(
            terrain_types, terrain_confidence, friction_coefficients
        )
        
        # HA-Teacher update
        self._robot.energy_2d = self.ha_teacher.update(self._torque_optimizer.tracking_error)

        # HA-Teacher action
        ha_action, dwell_flag = self.ha_teacher.get_action()

        self._terminal_stance_ddq, action_mode = self.coordinator.get_terminal_action(
            hp_action=hp_action,
            ha_action=ha_action,
            plant_state=self._torque_optimizer.tracking_error,
            safety_subset=self.ha_teacher.safety_subset,
            dwell_flag=dwell_flag
        )

        # Action mode indices
        hp_indices = torch.argwhere(action_mode == ActionMode.STUDENT.value).squeeze(-1)
        ha_indices = torch.argwhere(action_mode == ActionMode.TEACHER.value).squeeze(-1)
        hp_motor_action = None
        ha_motor_action = None

        # HP-Student in Control
        if len(hp_indices) > 0:
            hp_motor_action, self._desired_acc[hp_indices], self._solved_acc[hp_indices], \
                self._qp_cost[hp_indices], self._num_clips[hp_indices] = self._torque_optimizer.get_action(self._gait_generator.desired_contact_state, self._swing_leg_controller.desired_foot_positions, torch.zeros_like(self._gait_generator.desired_contact_state[:, :6]), desired_contact_state, desired_foot_positions, torch.zeros_like(desired_contact_state[:, :6]), 
                self._gait_generator.desired_contact_state[hp_indices],
                self._swing_leg_controller.desired_foot_positions[hp_indices],
                self._terminal_stance_ddq[hp_indices])
            self._student_activation_times[hp_indices] += 1
            self.robot.set_robot_base_color(color=(0, 0, 1.), env_ids=hp_indices)  # Display Blue
            nominal_actions[hp_indices] = drl_action  # Nominal actions for HP-student

        # HA-Teacher in Control
        if len(ha_indices) > 0:
            ha_motor_action, self._desired_acc[ha_indices], self._solved_acc[ha_indices], \
                self._qp_cost[ha_indices], self._num_clips[ha_indices] = self._torque_optimizer.get_action(self._gait_generator.desired_contact_state, self._swing_leg_controller.desired_foot_positions, torch.zeros_like(self._gait_generator.desired_contact_state[:, :6]), desired_contact_state, desired_foot_positions, torch.zeros_like(desired_contact_state[:, :6]), 
                self._gait_generator.desired_contact_state[ha_indices],
                self._swing_leg_controller.desired_foot_positions[ha_indices],
                self._terminal_stance_ddq[ha_indices],
                gravity_frame=True)
            self._teacher_activation_times[ha_indices] += 1
            self.robot.set_robot_base_color(color=(1., 0, 0), env_ids=ha_indices)  # Display Red
            # Nominal actions for HP-student
            nominal_actions[ha_indices] = (ha_action[ha_indices] - phy_action_res[ha_indices]) / weights

        # Unknown Action Mode
        if len(hp_indices) == 0 and len(ha_indices) == 0:
            raise RuntimeError(f"Unrecognized Action Mode: {action_mode}")

        # Add both HP-Student and HA-Teacher Motor Action
        motor_action = concatenate_motor_actions(command1=hp_motor_action, indices1=hp_indices,
                                                 command2=ha_motor_action, indices2=ha_indices)

        # Step the motor action
        self._robot.step(motor_action)

        # Get observations
        self._obs_buf = self._get_observations()
        self._privileged_obs_buf = self._get_privileged_observations()

        # Delta action for smoothness
        delta_action = torch.sqrt(torch.sum(
            torch.square(self._last_terminal_action - self._terminal_stance_ddq), dim=1))

        # Error in next step
        err_next = self._torque_optimizer.tracking_error

        # Dones or not
        is_done, is_fail = self._is_done()
        dones = torch.logical_or(dones, is_done)

        # Get Total reward
        rewards = self._go2_reward.compute_reward(err_prev, err_next, is_fail, delta_action)

        # Update episode statistics
        self._episode_length_buf += 1
        self._episode_rewards += rewards

        # Check if episode is done
        if torch.any(dones):
            self._episode_count[dones] += 1
            self._episode_successes[dones] = torch.logical_not(is_fail[dones])

        return self._obs_buf, rewards, dones, self._extras

    def _get_observations(self):
        """Get observations from the robot."""
        # Get base observations
        base_pos = self._robot.base_position
        base_vel = self._robot.base_velocity_body_frame
        base_ang_vel = self._robot.base_angular_velocity_body_frame
        base_orientation = self._robot.base_orientation_rpy

        # Get motor observations
        motor_pos = self._robot.motor_positions
        motor_vel = self._robot.motor_velocities

        # Get foot contact observations
        foot_contact = self._robot.foot_contact_forces

        # Get tactile sensor observations
        tactile_obs = self._robot.tactile_sensor.get_terrain_observations()

        # Concatenate all observations
        obs = torch.cat([
            base_pos, base_vel, base_ang_vel, base_orientation,
            motor_pos, motor_vel, foot_contact, tactile_obs
        ], dim=-1)

        return obs

    def get_observations(self):
        return self._get_observations()

    def _get_privileged_observations(self):
        """Get privileged observations."""
        return self._get_observations()

    def get_privileged_observations(self):
        return self._get_privileged_observations()

    def _episode_terminated(self):
        return self._episode_length_buf >= self._config.max_episode_length

    def update_episodic_statistics(self, env_ids):
        # Episode count
        self._episode_count[env_ids] += 1

        # Episode rewards
        self._episode_rewards[env_ids] = 0

        # Episode successes
        self._episode_successes[env_ids] = False

        # Episode length
        self._episode_length_buf[env_ids] = 0

    def _is_terminate(self):
        """Check if episode should be terminated."""
        # Check if robot has fallen
        base_height = self._robot.base_position[:, 2]
        fallen = base_height < 0.1

        # Check if robot has reached goal
        goal_distance = torch.norm(self._robot.base_position[:, :2] - self._goal_position, dim=-1)
        reached_goal = goal_distance < 0.5

        # Check if episode length exceeded
        episode_timeout = self._episode_length_buf >= self._config.max_episode_length

        return fallen, reached_goal, episode_timeout

    def _is_fail(self):
        """Check if episode has failed."""
        fallen, _, _ = self._is_terminate()
        return fallen

    def _is_done(self):
        """Check if episode is done."""
        fallen, reached_goal, episode_timeout = self._is_terminate()
        done = fallen | reached_goal | episode_timeout
        return done, fallen

    @property
    def device(self):
        return self._device

    @property
    def robot(self):
        return self._robot

    @property
    def gait_generator(self):
        return self._gait_generator

    @property
    def torque_optimizer(self):
        return self._torque_optimizer

    @property
    def goal_position(self):
        return self._goal_position

    @property
    def goal_distance(self):
        return torch.norm(self._robot.base_position[:, :2] - self._goal_position, dim=-1)

    @property
    def goal_yaw_difference(self):
        goal_yaw = torch.atan2(self._goal_position[:, 1] - self._robot.base_position[:, 1],
                              self._goal_position[:, 0] - self._robot.base_position[:, 0])
        current_yaw = self._robot.base_orientation_rpy[:, 2]
        yaw_diff = goal_yaw - current_yaw
        yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))
        return yaw_diff

    @property
    def planner(self):
        return self._planner

    @property
    def action_space(self):
        return (self._action_lb, self._action_ub)

    @property
    def observation_space(self):
        return (self._observation_lb, self._observation_ub)

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def num_obs(self):
        return self._config.obs_dim

    @property
    def num_privileged_obs(self):
        return self._config.obs_dim

    @property
    def num_actions(self):
        return self._config.act_dim

    @property
    def max_episode_length(self):
        return self._config.max_episode_length

    @property
    def episode_length_buf(self):
        return self._episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, new_length: torch.Tensor):
        self._episode_length_buf = new_length
