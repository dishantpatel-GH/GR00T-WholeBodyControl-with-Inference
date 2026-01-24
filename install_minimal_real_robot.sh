#!/bin/bash
# =============================================================================
# Minimal Installation Script for Real Robot Control
# =============================================================================
# This script installs ONLY the dependencies needed to run:
#   - scripts/run_gr00t_inference_policy.py
#   - gr00t_wbc/control/main/teleop/run_g1_control_loop.py
#
# This is a lightweight installation for running on real robot hardware.
# For full development environment, use the standard installation.
# =============================================================================

set -e  # Exit immediately on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  GR00T Real Robot Minimal Installation${NC}"
echo -e "${GREEN}================================================${NC}"

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ENV_NAME="${GR00T_ENV_NAME:-gr00t_robot}"
PYTHON_VERSION="3.10"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------------------------------------------------------
# Step 0: Check if miniforge/conda is installed
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[Step 0/6]${NC} Checking conda/mamba installation..."

if ! command -v mamba &> /dev/null && ! command -v conda &> /dev/null; then
    echo -e "${YELLOW}Conda/Mamba not found. Installing Miniforge...${NC}"
    
    # Detect architecture
    ARCH=$(uname -m)
    if [ "$ARCH" = "x86_64" ]; then
        MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    elif [ "$ARCH" = "aarch64" ]; then
        MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"
    else
        echo -e "${RED}Unsupported architecture: $ARCH${NC}"
        exit 1
    fi
    
    curl -L -O "$MINIFORGE_URL"
    bash Miniforge3-Linux-*.sh -b -p "$HOME/miniforge3"
    rm Miniforge3-Linux-*.sh
    
    # Initialize conda
    eval "$("$HOME/miniforge3/bin/conda" shell.bash hook)"
    conda init bash
    
    echo -e "${GREEN}Miniforge installed. Please restart your terminal and run this script again.${NC}"
    exit 0
fi

# Prefer mamba if available
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
else
    CONDA_CMD="conda"
fi

echo -e "${GREEN}Using: $CONDA_CMD${NC}"

# -----------------------------------------------------------------------------
# Step 1: Create conda environment
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[Step 1/6]${NC} Creating conda environment: ${ENV_NAME}..."

if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}Environment '${ENV_NAME}' already exists.${NC}"
    read -p "Do you want to recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n "$ENV_NAME" -y
        $CONDA_CMD create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    fi
else
    $CONDA_CMD create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

# Activate environment
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo -e "${GREEN}Environment activated: $CONDA_DEFAULT_ENV${NC}"

# -----------------------------------------------------------------------------
# Step 2: Configure channels and install ROS2 Humble (minimal)
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[Step 2/6]${NC} Installing ROS2 Humble (minimal)..."

# Add channels
conda config --env --add channels conda-forge
conda config --env --add channels robostack-staging

# Remove defaults to avoid conflicts
conda config --env --remove channels defaults 2>/dev/null || true

# Install ROS2 base (not full desktop - we don't need rviz, rqt, etc.)
$CONDA_CMD install -y ros-humble-ros-base

# Install additional ROS packages needed
$CONDA_CMD install -y \
    ros-humble-std-msgs \
    ros-humble-sensor-msgs \
    ros-humble-std-srvs

echo -e "${GREEN}ROS2 Humble base installed${NC}"

# -----------------------------------------------------------------------------
# Step 3: Install CycloneDDS and RMW implementation
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[Step 3/6]${NC} Installing CycloneDDS..."

# Install CycloneDDS RMW implementation via conda (for ROS2)
$CONDA_CMD install -y ros-humble-rmw-cyclonedds-cpp

# Install Python CycloneDDS bindings via pip
pip install cyclonedds==0.10.2

echo -e "${GREEN}CycloneDDS installed${NC}"

# -----------------------------------------------------------------------------
# Step 4: Install pip dependencies
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[Step 4/6]${NC} Installing Python dependencies (minimal)..."

pip install --upgrade pip

# Install from minimal requirements file
pip install -r "${PROJECT_DIR}/requirements_minimal_real_robot.txt"

echo -e "${GREEN}Python dependencies installed${NC}"

# -----------------------------------------------------------------------------
# Step 5: Install Unitree SDK
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[Step 5/6]${NC} Installing Unitree SDK..."

pip install -e "${PROJECT_DIR}/external_dependencies/unitree_sdk2_python" --no-deps

echo -e "${GREEN}Unitree SDK installed${NC}"

# -----------------------------------------------------------------------------
# Step 6: Install gr00t_wbc (editable, minimal)
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[Step 6/6]${NC} Installing gr00t_wbc package (minimal mode)..."

# Install in editable mode without full dependencies
pip install -e "${PROJECT_DIR}" --no-deps

echo -e "${GREEN}gr00t_wbc installed${NC}"

# -----------------------------------------------------------------------------
# Setup ROS environment
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Configuring ROS environment...${NC}"

# Source ROS setup
source "$CONDA_PREFIX/setup.bash"

# Add ROS setup to activation script
ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$ACTIVATE_DIR"
cat > "$ACTIVATE_DIR/ros_setup.sh" << 'EOF'
#!/bin/bash
source "$CONDA_PREFIX/setup.bash"
export ROS_LOCALHOST_ONLY=1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
EOF

# Create deactivate script
DEACTIVATE_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"
mkdir -p "$DEACTIVATE_DIR"
cat > "$DEACTIVATE_DIR/ros_cleanup.sh" << 'EOF'
#!/bin/bash
unset ROS_LOCALHOST_ONLY
unset RMW_IMPLEMENTATION
EOF

echo -e "${GREEN}ROS environment configured${NC}"

# -----------------------------------------------------------------------------
# Verify installation
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Verifying installation...${NC}"

echo -n "  Python: "
python --version

echo -n "  NumPy: "
python -c "import numpy; print(numpy.__version__)"

echo -n "  PyTorch: "
python -c "import torch; print(torch.__version__)"

echo -n "  Pinocchio: "
python -c "import pinocchio; print(pinocchio.__version__)"

echo -n "  ROS2: "
python -c "import rclpy; print('OK')"

echo -n "  CycloneDDS: "
python -c "import cyclonedds; print('OK')"

echo -n "  ZMQ: "
python -c "import zmq; print(zmq.__version__)"

echo -n "  gr00t_wbc: "
python -c "import gr00t_wbc; print('OK')"

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}  Installation Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo -e ""
echo -e "To activate the environment:"
echo -e "  ${YELLOW}conda activate ${ENV_NAME}${NC}"
echo -e ""
echo -e "To run the control loop:"
echo -e "  ${YELLOW}python gr00t_wbc/control/main/teleop/run_g1_control_loop.py${NC}"
echo -e ""
echo -e "To run the inference policy:"
echo -e "  ${YELLOW}python scripts/run_gr00t_inference_policy.py --model_host <server_ip>${NC}"
echo -e ""
echo -e "${YELLOW}Note:${NC} The GR00T model server should run on a separate GPU machine."
echo -e "      Camera server should also run separately on the robot's Orin."
