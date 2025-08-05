# Tactile Sensing and Adaptive Gait System

This document describes the tactile sensing and adaptive gait system implemented for the Isaac-Wild-Go2 quadruped robot.

## Overview

The tactile sensing system enables the robot to detect terrain properties through contact forces and adapt its gait accordingly for improved efficiency and safety across different surface types.

## System Architecture

### 1. Tactile Sensor Module (`TactileSensor`)

**Location**: `src/envs/robots/modules/sensor/tactile_sensor.py`

The tactile sensor processes contact data from the robot's feet to classify terrain types and estimate friction coefficients.

#### Key Features:
- **Terrain Classification**: Detects 8 different terrain types (concrete, sand, ice, grass, rock, mud, snow, metal)
- **Friction Estimation**: Estimates friction coefficients based on contact characteristics
- **Confidence Scoring**: Provides confidence levels for terrain detection
- **Moving Average**: Uses historical data for stable detection

#### Terrain Types and Properties:
| Terrain | Friction | Stability | Gait Adaptation |
|---------|----------|-----------|-----------------|
| Concrete | 0.8 | High | Standard gait |
| Sand | 0.6 | Medium | Higher steps, slower frequency |
| Ice | 0.1 | Low | Lower steps, much slower frequency |
| Grass | 0.4 | Medium | Moderate adaptation |
| Rock | 0.9 | High | Higher steps for uneven terrain |
| Mud | 0.3 | Low | Higher steps, longer stance |
| Snow | 0.2 | Low | Moderate steps, conservative gait |
| Metal | 0.5 | Medium | Standard gait |

### 2. Adaptive Gait Controller (`AdaptiveGaitController`)

**Location**: `src/envs/robots/modules/controller/adaptive_gait_controller.py`

The adaptive gait controller modifies gait parameters based on terrain detection to optimize locomotion efficiency.

#### Gait Parameters:
- **Step Height**: Vertical clearance during swing phase
- **Step Frequency**: Rate of leg movements
- **Stance Duration**: Time spent in contact with ground
- **Swing Duration**: Time spent in air
- **Foot Clearance**: Minimum height above ground
- **Stability Margin**: Safety buffer for stability

#### Terrain-Specific Adaptations:

**Ice/Snow (Low Friction)**:
- Reduced step frequency (1.2-1.3 Hz)
- Increased stance duration (0.45-0.5s)
- Lower step height (0.06-0.08m)
- Higher stability margin (0.17-0.2)

**Sand/Mud (Medium Friction)**:
- Moderate step frequency (1.4-1.5 Hz)
- Longer stance duration (0.4-0.5s)
- Higher step height (0.12-0.14m)
- Medium stability margin (0.15-0.16)

**Rock/Concrete (High Friction)**:
- Higher step frequency (1.6-2.0 Hz)
- Shorter stance duration (0.3-0.45s)
- Variable step height (0.08-0.15m)
- Lower stability margin (0.1-0.18)

## Integration with Existing System

### 1. Robot Integration

The tactile sensor is integrated into the robot class (`src/envs/robots/robot.py`):

```python
# Initialize tactile sensor
self._tactile_sensor = TactileSensor(num_envs=self._num_envs, device=self._device)

# Update tactile sensor in physics step
def _update_tactile_sensor(self):
    self._tactile_sensor.update(
        robot_contact_forces=self.foot_contact_forces,
        robot_contact_torques=torch.zeros_like(self.foot_contact_forces),
        foot_positions=self.foot_positions_in_world_frame,
        foot_velocities=self.foot_velocities_in_world_frame,
        foot_contacts=self.foot_contacts
    )
```

### 2. Environment Integration

The adaptive gait controller is integrated into the environment (`src/envs/go2_wild_env.py`):

```python
# Initialize adaptive gait controller
self._adaptive_gait_controller = AdaptiveGaitController(num_envs=self._num_envs, device=self._device)

# Update gait parameters in step function
adapted_gait_params = self._adaptive_gait_controller.update_gait_parameters(
    terrain_types, terrain_confidence, friction_coefficients
)
```

### 3. Observation Space Extension

The observation space is extended to include tactile sensor data:

```python
# Get tactile sensor observations
terrain_obs = self._robot.tactile_sensor.get_terrain_observations()  # 12 values (4 feet × 3 values)
contact_obs = self._robot.tactile_sensor.get_foot_contact_observations()  # 12 values (4 feet × 3 values)

# Combine with existing observations
obs = torch.concatenate((distance_obs, yaw_diff_obs, robot_obs, terrain_obs, contact_obs), dim=1)
```

## Reward System

### New Reward Functions

Two new reward functions are added to encourage terrain-appropriate behavior:

#### 1. Terrain Adaptation Reward (`terrain_adaptation_reward`)
- Rewards successful terrain detection
- Encourages appropriate gait adaptation
- Based on terrain confidence and adaptation success

#### 2. Terrain Efficiency Reward (`terrain_efficiency_reward`)
- Rewards efficient locomotion on different terrains
- Ice/Snow: Rewards conservative, careful movement
- Rock/Concrete: Rewards faster, more aggressive movement
- Sand/Mud: Rewards balanced, moderate movement

## Configuration

The system is configured through `src/configs/tactile_sensor_config.py`:

```python
# Tactile sensor parameters
config.force_threshold = 5.0  # Minimum force to consider contact
config.slip_threshold = 0.1   # Slip velocity threshold
config.area_threshold = 0.01   # Minimum contact area
config.history_length = 10     # Number of timesteps for moving average

# Adaptive gait parameters
config.adaptation_rate = 0.1  # How quickly to adapt gait parameters
config.stability_threshold = 0.7  # Minimum confidence for terrain adaptation
```

## Usage Examples

### Running the Test Script

```bash
python -m src.scripts.test_tactile_sensing
```

### Training with Tactile Sensing

The tactile sensing system is automatically integrated into the training process. The observation space is extended to include terrain and contact data, and the reward system includes terrain-aware rewards.

### Real-time Terrain Detection

During operation, the system continuously:
1. Monitors contact forces and velocities
2. Classifies terrain types
3. Estimates friction coefficients
4. Adapts gait parameters
5. Provides terrain-aware rewards

## Performance Benefits

### 1. Safety Improvements
- **Ice/Snow**: Conservative gait prevents slipping
- **Rock**: Higher step clearance prevents tripping
- **Sand/Mud**: Longer stance duration improves stability

### 2. Efficiency Gains
- **Concrete**: Faster, more efficient movement
- **Grass**: Balanced gait for variable conditions
- **Metal**: Standard gait for predictable surfaces

### 3. Energy Optimization
- Terrain-appropriate step frequencies reduce energy consumption
- Optimized stance/swing ratios improve efficiency
- Reduced slipping and stumbling minimize energy waste

## Future Enhancements

1. **Advanced Terrain Classification**: Machine learning-based terrain recognition
2. **Dynamic Friction Estimation**: Real-time friction coefficient updates
3. **Multi-modal Sensing**: Integration with vision and proprioceptive sensors
4. **Learning-based Adaptation**: End-to-end learning of terrain-specific gaits
5. **Real Robot Deployment**: Hardware integration for real tactile sensors

## Troubleshooting

### Common Issues

1. **Low Terrain Confidence**: Check contact force thresholds
2. **Incorrect Terrain Classification**: Verify slip velocity calculations
3. **Gait Adaptation Issues**: Check adaptation rate and stability threshold
4. **Observation Space Mismatch**: Ensure observation dimensions match network input

### Debugging

Use the test script to verify system functionality:
```bash
python -m src.scripts.test_tactile_sensing
```

Check tactile sensor outputs during training:
```python
# Print terrain detection results
print(f"Terrain types: {robot.tactile_sensor.detected_terrain_types}")
print(f"Friction coefficients: {robot.tactile_sensor.estimated_friction}")
print(f"Confidence levels: {robot.tactile_sensor.terrain_confidence}")
``` 