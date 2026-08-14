#!/bin/bash
# =============================================================================
#  scripts/slurm_resume.sh
#  UA-LDG: Resume training from best_model.pt checkpoint.
#
#  Use this after a run that hit the epoch limit before the LR scheduler fired.
#  Resumes model + optimiser state; LR scheduler restarts with full patience.
#
#  Submit: sbatch scripts/slurm_resume.sh
# =============================================================================

#SBATCH --job-name=ua_ldg_resume
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:h100-80:2
#SBATCH --mem=64GB
#SBATCH --time=28:00:00
#SBATCH --output=outputs/logs/train_%j.out
#SBATCH --error=outputs/logs/train_%j.err
#SBATCH -A cis250046p

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  UA-LDG Resume Job"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Node       : $(hostname)"
echo "  GPUs       : $SLURM_GPUS_ON_NODE"
echo "  CPUs/task  : $SLURM_CPUS_PER_TASK"
echo "  Resuming   : outputs/checkpoints/best_model.pt"
echo "  Start      : $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

module purge
module load anaconda3/2024.10-1
module load cuda

source "/ocean/projects/cis250046p/oali2/DeepLearningProject/XrayGNN/venv/bin/activate"

echo "Python  : $(python3 --version)"
echo "PyTorch : $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA    : $(python3 -c 'import torch; print(torch.version.cuda)')"
python3 -c "import torch; [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
echo ""

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p outputs/logs outputs/checkpoints outputs/results

echo "── Resuming training ──────────────────────────────"

python3 scripts/train.py \
    --config configs/default.yaml \
    --config configs/bridges2.yaml \
    --resume outputs/checkpoints/best_model.pt \
    training.epochs=50 \
    data.num_workers=$SLURM_CPUS_PER_TASK

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Training complete at $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
