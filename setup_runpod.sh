#!/bin/

# RunPod Setup Script for Isaac-Wild-Go2 with Tactile Sensing
# Run this script in a new RunPod instance to set up all dependencies

set -e  # Exit on any error

echo "🚀 Setting up Isaac-Wild-Go2 with Tactile Sensing..."

# Update system packages
echo "📦 Updating system packages..."
apt update
apt install -y build-essential cmake git wget curl
apt install -y libboost-all-dev liblcm-dev
apt install -y libgl1-mesa-glx libglib2.0-0
apt install -y python3-pip python3-dev

# Create conda environment (if not exists)
echo "🐍 Setting up Python environment..."
if ! conda env list | grep -q "isaac_py38"; then
    conda create -n isaac_py38 python=3.8 -y
fi

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaac_py38

# Install Python packages
echo "📚 Installing Python packages..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy matplotlib
pip install opencv-python
pip install open3d
pip install ml-collections
pip install absl-py
pip install gym==0.21.0

# Install IsaacGym (you'll need to download this manually)
echo "🎮 IsaacGym setup required:"
echo "1. Download IsaacGym from NVIDIA developer portal"
echo "2. Extract to /workspace/isaacgym"
echo "3. Run: cd /workspace/isaacgym/python && pip install -e ."

# Install RSL-RL
echo "🤖 Installing RSL-RL..."
cd /workspace
if [ ! -d "rsl_rl" ]; then
    git clone https://github.com/leggedrobotics/rsl_rl.git
fi
cd rsl_rl
pip install -e .

# Clone your project
echo "📁 Cloning your project..."
cd /workspace
if [ ! -d "isaac-wild-go-trial-" ]; then
    git clone https://github.com/jasmehsoccer/isaac-wild-go-trial-.git
fi

# Build go2_sdk
echo "🔧 Building go2_sdk..."
cd isaac-wild-go-trial-/extern/go2_sdk
mkdir -p build && cd build
cmake ..
make -j$(nproc)
mv go2_interface* ../../../

# Set up environment variables
echo "🔧 Setting up environment variables..."
echo 'export PYTHONPATH=/workspace/isaac-wild-go-trial-:/workspace/isaac-wild-go-trial-/extern/go2_sdk:$PYTHONPATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/workspace/isaac-wild-go-trial-:$LD_LIBRARY_PATH' >> ~/.bashrc

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Download and install IsaacGym"
echo "2. Run: conda activate isaac_py38"
echo "3. Run: cd /workspace/isaac-wild-go-trial-"
echo "4. Run: PYTHONPATH=.:extern/go2_sdk python src/scripts/train.py --config=src/configs/wild_env_config.py --num_envs=1 --use_gpu=True --show_gui=False --record_videos=True --logdir=logs --experiment_name=tactile_sensing_training"
#!/bin/bash

# RunPod Setup Script for Isaac-Wild-Go2 with Tactile Sensing
# Run this script in a new RunPod instance to set up all dependencies

set -e  # Exit on any error

echo "🚀 Setting up Isaac-Wild-Go2 with Tactile Sensing..."

# Update system packages (no sudo needed in RunPod)
echo "📦 Updating system packages..."
apt update
apt install -y build-essential cmake git wget curl
apt install -y libboost-all-dev liblcm-dev
apt install -y libgl1-mesa-glx libglib2.0-0
apt install -y python3-pip python3-dev

# Create conda environment (if not exists)
echo "🐍 Setting up Python environment..."
if ! conda env list | grep -q "isaac_py38"; then
    conda create -n isaac_py38 python=3.8 -y
fi

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaac_py38

# Install Python packages
echo "📚 Installing Python packages..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy matplotlib
pip install opencv-python
pip install open3d
pip install ml-collections
pip install absl-py
pip install gym==0.21.0

# Install IsaacGym (you'll need to download this manually)
echo "🎮 IsaacGym setup required:"
echo "1. Download IsaacGym from NVIDIA developer portal"
echo "2. Extract to /workspace/isaacgym"
echo "3. Run: cd /workspace/isaacgym/python && pip install -e ."

# Install RSL-RL
echo "🤖 Installing RSL-RL..."
cd /workspace
if [ ! -d "rsl_rl" ]; then
    git clone https://github.com/leggedrobotics/rsl_rl.git
fi
cd rsl_rl
pip install -e .

# Clone your project
echo "📁 Cloning your project..."
cd /workspace
if [ ! -d "isaac-wild-go-trial-" ]; then
    git clone https://github.com/jasmehsoccer/isaac-wild-go-trial-.git
fi

# Build go2_sdk
echo "🔧 Building go2_sdk..."
cd isaac-wild-go-trial-/extern/go2_sdk
mkdir -p build && cd build
cmake ..
make -j$(nproc)
mv go2_interface* ../../../

# Set up environment variables
echo "🔧 Setting up environment variables..."
echo 'export PYTHONPATH=/workspace/isaac-wild-go-trial-:/workspace/isaac-wild-go-trial-/extern/go2_sdk:$PYTHONPATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/workspace/isaac-wild-go-trial-:$LD_LIBRARY_PATH' >> ~/.bashrc

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Download and install IsaacGym"
echo "2. Run: conda activate isaac_py38"
echo "3. Run: cd /workspace/isaac-wild-go-trial-"
echo "4. Run: PYTHONPATH=.:extern/go2_sdk python src/scripts/train.py --config=src/configs/wild_env_config.py --num_envs=1 --use_gpu=True --show_gui=False --record_videos=True --logdir=logs --experiment_name=tactile_sensing_training"
#!/bin/bash

# RunPod Setup Script for Isaac-Wild-Go2 with Tactile Sensing
# Run this script in a new RunPod instance to set up all dependencies

set -e  # Exit on any error

echo "🚀 Setting up Isaac-Wild-Go2 with Tactile Sensing..."

# Update system packages (no sudo needed in RunPod)
echo "📦 Updating system packages..."
apt update
apt install -y build-essential cmake git wget curl
apt install -y libboost-all-dev liblcm-dev
apt install -y libgl1-mesa-glx libglib2.0-0
apt install -y python3-pip python3-dev

# Create conda environment (if not exists)
echo "🐍 Setting up Python environment..."
if ! conda env list | grep -q "isaac_py38"; then
    conda create -n isaac_py38 python=3.8 -y
fi

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaac_py38

# Install Python packages
echo "📚 Installing Python packages..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy matplotlib
pip install opencv-python
pip install open3d
pip install ml-collections
pip install absl-py
pip install gym==0.21.0

# Install IsaacGym (you'll need to download this manually)
echo "🎮 IsaacGym setup required:"
echo "1. Download IsaacGym from NVIDIA developer portal"
echo "2. Extract to /workspace/isaacgym"
echo "3. Run: cd /workspace/isaacgym/python && pip install -e ."

# Install RSL-RL
echo "🤖 Installing RSL-RL..."
cd /workspace
if [ ! -d "rsl_rl" ]; then
    git clone https://github.com/leggedrobotics/rsl_rl.git
fi
cd rsl_rl
pip install -e .

# Clone your project
echo "📁 Cloning your project..."
cd /workspace
if [ ! -d "isaac-wild-go-trial-" ]; then
    git clone https://github.com/jasmehsoccer/isaac-wild-go-trial-.git
fi

# Build go2_sdk
echo "🔧 Building go2_sdk..."
cd isaac-wild-go-trial-/extern/go2_sdk
mkdir -p build && cd build
cmake ..
make -j$(nproc)
mv go2_interface* ../../../

# Set up environment variables
echo "🔧 Setting up environment variables..."
echo 'export PYTHONPATH=/workspace/isaac-wild-go-trial-:/workspace/isaac-wild-go-trial-/extern/go2_sdk:$PYTHONPATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/workspace/isaac-wild-go-trial-:$LD_LIBRARY_PATH' >> ~/.bashrc

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Download and install IsaacGym"
echo "2. Run: conda activate isaac_py38"
echo "3. Run: cd /workspace/isaac-wild-go-trial-"
echo "4. Run: PYTHONPATH=.:extern/go2_sdk python src/scripts/train.py --config=src/configs/wild_env_config.py --num_envs=1 --use_gpu=True --show_gui=False --record_videos=True --logdir=logs --experiment_name=tactile_sensing_training"
