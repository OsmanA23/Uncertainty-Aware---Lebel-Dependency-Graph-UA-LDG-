#!/bin/bash
# =============================================================================
#  scripts/slurm_interactive.sh
#  UA-LDG: Launch an interactive GPU session for debugging on Bridges-2
#
#  Use this for:
#    - Running quick sanity checks on a GPU
#    - Debugging forward/backward pass issues
#    - Testing the data loader
#    - Running pytest on GPU
#
#  Do NOT use for full training runs — those go through slurm_train.sh
#
#  Usage:
#    bash scripts/slurm_interactive.sh
#
#  This will drop you into an interactive shell on a GPU node.
#  Your environment will be pre-loaded.
# =============================================================================

echo ""
echo "Requesting interactive GPU session on Bridges-2..."
echo "This may wait in queue if all GPUs are busy."
echo ""
echo "Requesting: 1× V100, 16GB RAM, 4 CPUs, 2 hours"
echo ""

# Using account cis250046p
interact \
    --partition GPU-shared \
    --gres gpu:v100-32:1 \
    --ntasks-per-node 1 \
    --cpus-per-task 4 \
    --mem 16GB \
    --time 02:00:00 \
    -A cis250046p \
    --command "
        module purge &&
        module load anaconda3/2024.10-1 &&
        module load cuda &&
        source /ocean/projects/cis250046p/oali2/DeepLearningProject/XrayGNN/venv/bin/activate &&
        echo '── UA-LDG interactive session ready ──' &&
        echo 'Python:  '$(python3 --version) &&
        echo 'PyTorch: '$(python3 -c 'import torch; print(torch.__version__)') &&
        echo 'GPU:     '$(python3 -c 'import torch; print(torch.cuda.get_device_name(0))') &&
        bash
    "
