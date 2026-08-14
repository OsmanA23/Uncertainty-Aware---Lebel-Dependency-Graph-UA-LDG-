#!/bin/bash
# =============================================================================
#  scripts/slurm_ablation.sh
#  UA-LDG: Ablation study — 3 variants run as a SLURM array job.
#
#  Array index → ablation variant:
#    0: Ablation A — DenseNet-only    (no graph)
#    1: Ablation B — Fixed GCN        (plain co-occurrence weights, no gate)
#    2: Ablation C — UA-GCN, no gate  (Beta-Binomial edges, no attention gate)
#
#  All three variants use the same tuned hyperparameters as the full model
#  (default.yaml + bridges2.yaml) — only the model component differs.
#
#  Submit all 3 in parallel:  sbatch scripts/slurm_ablation.sh
#  Submit one variant:        sbatch --array=0 scripts/slurm_ablation.sh
#  Monitor:                   squeue -u $USER
# =============================================================================

#SBATCH --job-name=ua_ldg_ablation
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:h100-80:2
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --array=0-2
#SBATCH --output=outputs/logs/ablation_%A_%a.out
#SBATCH --error=outputs/logs/ablation_%A_%a.err
#SBATCH -A cis250046p

# ── Map array index to ablation config ───────────────────────────────────────
case $SLURM_ARRAY_TASK_ID in
    0)  ABLATION_NAME="densenet_only"
        ABLATION_CONFIG="configs/ablation_densenet_only.yaml"
        ;;
    1)  ABLATION_NAME="fixed_gcn"
        ABLATION_CONFIG="configs/ablation_fixed_gcn.yaml"
        ;;
    2)  ABLATION_NAME="no_gate"
        ABLATION_CONFIG="configs/ablation_no_gate.yaml"
        ;;
esac

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  UA-LDG Ablation: $ABLATION_NAME"
echo "  Array task : $SLURM_ARRAY_TASK_ID / ${SLURM_ARRAY_JOB_ID}"
echo "  Config     : $ABLATION_CONFIG"
echo "  Node       : $(hostname)"
echo "  GPUs       : $SLURM_GPUS_ON_NODE"
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

mkdir -p outputs/logs \
         "outputs/ablations/${ABLATION_NAME}/checkpoints" \
         "outputs/ablations/${ABLATION_NAME}/logs" \
         "outputs/ablations/${ABLATION_NAME}/results"

echo "── Starting ablation: $ABLATION_NAME ─────────────"

python3 scripts/train.py \
    --config configs/default.yaml \
    --config configs/bridges2.yaml \
    --config "$ABLATION_CONFIG" \
    data.num_workers=$SLURM_CPUS_PER_TASK

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Ablation $ABLATION_NAME complete at $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
