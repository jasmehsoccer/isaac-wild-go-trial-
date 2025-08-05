# Manual Fix for NumPy Compatibility Issue

## The Problem
IsaacGym uses deprecated `np.float` which was removed in newer NumPy versions. This causes the error:
```
AttributeError: module 'numpy' has no attribute 'float'
```

## Quick Fix (Run on RunPod)

### Option 1: Use the fix script
```bash
python fix_numpy_compatibility.py
```

### Option 2: Manual fix
```bash
# Find the torch_utils.py file
find /workspace -name "torch_utils.py" 2>/dev/null

# Edit the file (replace PATH with the actual path found above)
sed -i 's/np\.float/float/g' /workspace/isaacgym/python/isaacgym/torch_utils.py
```

### Option 3: Direct replacement
```bash
# Replace all instances of np.float with float
sed -i 's/np\.float/float/g' /workspace/isaacgym/python/isaacgym/torch_utils.py
```

## After the fix, run training:
```bash
python src/scripts/train.py --config=src/configs/wild_env_config.py --num_envs=1 --use_gpu=True --show_gui=False --record_videos=True --logdir=logs --experiment_name=tactile_sensing_training
```

## What this fixes:
- ✅ Replaces deprecated `np.float` with `float`
- ✅ Maintains compatibility with newer NumPy versions
- ✅ Allows IsaacGym to work properly
- ✅ Enables training with tactile sensing 