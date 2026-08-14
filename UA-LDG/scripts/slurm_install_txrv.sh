#!/bin/bash
# =============================================================================
#  scripts/slurm_install_txrv.sh
#  Install torchxrayvision into the project venv (CPU job, RM-shared).
#
#  Submit: sbatch scripts/slurm_install_txrv.sh
#  Then resubmit training: sbatch scripts/slurm_train.sh --resume outputs/checkpoints/best_model.pt
# =============================================================================

#SBATCH --job-name=install_txrv
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=00:30:00
#SBATCH --output=outputs/logs/install_txrv_%j.out
#SBATCH --error=outputs/logs/install_txrv_%j.err
#SBATCH -A cis250046p

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Installing torchxrayvision"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $(hostname)"
echo "  Start  : $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

module purge
module load anaconda3/2024.10-1

source "/ocean/projects/cis250046p/oali2/DeepLearningProject/XrayGNN/venv/bin/activate"
echo "Python : $(python3 --version)"

pip install torchxrayvision

echo ""
python3 -c "
import torchxrayvision as xrv
print('torchxrayvision version:', xrv.__version__)
model = xrv.models.DenseNet(weights='densenet121-res224-mimic_ch')
print('Weights loaded OK — densenet121-res224-mimic_ch')
"

echo ""
echo "  Done at $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
