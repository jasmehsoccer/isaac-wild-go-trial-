"""Train DDPG policy using implementation from RSL_RL."""
import logging
import pickle
import time

import numpy as np
import yaml
from absl import app
from absl import flags
# from absl import logging
from isaacgym import gymapi, gymutil
import torch
import cv2
import os

from datetime import datetime
import os

from ml_collections.config_flags import config_flags
from rsl_rl.runners import OnPolicyRunner

from src.configs.training import ddpg
from src.envs import env_wrappers

config_flags.DEFINE_config_file("config", "src/configs/wild_env_config.py", "experiment configuration.")
flags.DEFINE_integer("num_envs", 1, "number of parallel environments.")
flags.DEFINE_bool("use_gpu", True, "whether to use GPU.")
flags.DEFINE_bool("enable_ha_teacher", False, "whether to enable the HA-Teacher.")
flags.DEFINE_bool("enable_pusher", False, "whether to enable the robot pusher.")
flags.DEFINE_bool("use_real_robot", False, "whether to use real robot.")
flags.DEFINE_bool("show_gui", False, "whether to show GUI.")
flags.DEFINE_bool("record_videos", True, "whether to record training videos.")
flags.DEFINE_integer("video_interval", 1000, "record video every N iterations.")
flags.DEFINE_integer("video_length", 200, "number of steps to record in each video.")
flags.DEFINE_integer("video_fps", 30, "frames per second for video recording.")
flags.DEFINE_integer("video_width", 1280, "video width in pixels.")
flags.DEFINE_integer("video_height", 720, "video height in pixels.")
flags.DEFINE_string("logdir", "logs", "logdir.")
flags.DEFINE_string("load_checkpoint", None, "checkpoint to load.")
flags.DEFINE_string("experiment_name", None, "experimental name to set.")

FLAGS = flags.FLAGS


def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    return obj


def capture_frame_from_gym(env):
    """Capture a frame from IsaacGym using OpenCV."""
    # Get the current frame from IsaacGym
    env.robot._gym.step_graphics(env.robot._sim)
    env.robot._gym.draw_viewer(env.robot._viewer, env.robot._sim, True)
    
    # Capture the frame from the viewer
    frame = env.robot._gym.get_viewer_image(env.robot._viewer, gymapi.IMAGE_COLOR)
    
    # Convert to numpy array and reshape
    frame = np.array(frame, dtype=np.uint8)
    frame = frame.reshape(FLAGS.video_height, FLAGS.video_width, 4)  # RGBA
    frame = frame[:, :, :3]  # Convert RGBA to RGB
    
    # Flip vertically (IsaacGym coordinates are flipped)
    frame = cv2.flip(frame, 0)
    
    return frame


def record_training_video(env, logdir, iteration, video_length=200):
    """Record a video of the training progress."""
    video_dir = os.path.join(logdir, "videos")
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    
    video_filename = os.path.join(video_dir, f"training_iteration_{iteration:06d}.mp4")
    print(f"Recording video: {video_filename}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_filename, fourcc, FLAGS.video_fps, 
                                  (FLAGS.video_width, FLAGS.video_height))
    
    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for {video_filename}")
        return
    
    for step in range(video_length):
        if hasattr(env, 'action_space'):
            if isinstance(env.action_space, tuple):
                action_lb, action_ub = env.action_space
                actions = torch.rand(env.num_envs, action_lb.shape[0], device=env.device) * (action_ub - action_lb) + action_lb
            elif hasattr(env.action_space, 'sample'):
                actions = env.action_space.sample()
                if isinstance(actions, np.ndarray):
                    actions = torch.from_numpy(actions).to(env.device)
            else:
                actions = torch.rand(env.num_envs, env.num_actions, device=env.device) * 2 - 1
        else:
            actions = torch.rand(env.num_envs, env.num_actions, device=env.device) * 2 - 1
        
        obs, rewards, dones, infos = env.step(actions)
        
        frame = capture_frame_from_gym(env)
        video_writer.write(frame)
        
        time.sleep(1.0 / FLAGS.video_fps)
    
    video_writer.release()
    print(f"Video saved: {video_filename}")


def main(argv):

    del argv  # unused
    logging.disable(logging.CRITICAL)  # logging output

    device = "cuda" if FLAGS.use_gpu else "cpu"
    config = FLAGS.config

    # Experimental name set
    if FLAGS.experiment_name:
        config.training.runner.experiment_name = FLAGS.experiment_name

    logdir = os.path.join(FLAGS.logdir, config.training.runner.experiment_name,
                          datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    with open(os.path.join(logdir, "config.yaml"), "w", encoding="utf-8") as f:
        f.write(config.to_yaml())

    # Use DDPG
    config.training = ddpg.get_training_config()

    # HA-Teacher module
    if FLAGS.enable_ha_teacher:
        config.environment.ha_teacher.enable = True
        # config.environment.ha_teacher.chi = 0.1
        # config.environment.ha_teacher.tau = 100

    env = config.env_class(num_envs=FLAGS.num_envs,
                           device=device,
                           config=config.environment,
                           show_gui=FLAGS.show_gui,
                           use_real_robot=FLAGS.use_real_robot)
    # Robot pusher
    if FLAGS.enable_pusher:
        env._pusher.push_enable = True

    env = env_wrappers.RangeNormalize(env)

    # Set up camera for initial view
    mean_pos = torch.min(env.robot.base_position_world, dim=0)[0].cpu().numpy() + np.array([-2.5, -2.5, 2.5])
    target_pos = torch.mean(env.robot.base_position_world, dim=0).cpu().numpy() + np.array([0., 2., -0.5])
    cam_pos = gymapi.Vec3(*mean_pos)
    cam_target = gymapi.Vec3(*target_pos)
    env.robot._gym.viewer_camera_look_at(env.robot._viewer, None, cam_pos, cam_target)
    env.robot._gym.step_graphics(env.robot._sim)
    env.robot._gym.draw_viewer(env.robot._viewer, env.robot._sim, True)

    ddpg_runner = OnPolicyRunner(env, config.training, logdir, device=device)
    if FLAGS.load_checkpoint:
        ddpg_runner.load(FLAGS.load_checkpoint)
    
    # Custom training loop with video recording
    if FLAGS.record_videos:
        print(f"Training with video recording every {FLAGS.video_interval} iterations")
        print(f"Videos will be saved to: {os.path.join(logdir, 'videos')}")
        print(f"Video settings: {FLAGS.video_width}x{FLAGS.video_height} @ {FLAGS.video_fps} FPS")
        
        # Record initial video
        record_training_video(env, logdir, 0, FLAGS.video_length)
        
        # Custom training loop
        for iteration in range(config.training.runner.max_iterations):
            # Train for one iteration
            ddpg_runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
            
            # Record video periodically
            if (iteration + 1) % FLAGS.video_interval == 0:
                record_training_video(env, logdir, iteration + 1, FLAGS.video_length)
                
            # Print progress
            if (iteration + 1) % 100 == 0:
                print(f"Training progress: {iteration + 1}/{config.training.runner.max_iterations}")
    else:
        # Standard training without video recording
        ddpg_runner.learn(num_learning_iterations=config.training.runner.max_iterations,
                          init_at_random_ep_len=False)


if __name__ == "__main__":
    app.run(main)
