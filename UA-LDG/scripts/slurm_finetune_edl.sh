#!/bin/bash
# =============================================================================
#  scripts/slurm_finetune_edl.sh
#  UA-LDG: Quick EDL head fine-tune to validate learnable-β fix.
#
#  Loads best checkpoint, freezes backbone + GCN, trains only the new
#  EDL head for 2 epochs. Check ECE and p_hat distribution in the log
#  to decide whether to commit to a full retrain.
#
#  Submit: sbatch scripts/slurm_finetune_edl.sh
#  Monitor: tail -f outputs/logs/finetune_edl_JOBID.out
# =============================================================================

#SBATCH --job-name=ua_ldg_edl_ft
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=48:00:00
#SBATCH --output=outputs/logs/finetune_edl_%j.out
#SBATCH --error=outputs/logs/finetune_edl_%j.err
#SBATCH -A cis250046p

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  UA-LDG EDL Head Fine-tune (learnable-beta test)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Node    : $(hostname)"
echo "  GPU     : $SLURM_GPUS_ON_NODE"
echo "  Start   : $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

module purge
module load anaconda3/2024.10-1
module load cuda

source "/ocean/projects/cis250046p/oali2/DeepLearningProject/XrayGNN/venv/bin/activate"

echo "Python  : $(python3 --version)"
echo "PyTorch : $(python3 -c 'import torch; print(torch.__version__)')"
echo "Device  : CPU (RM-shared — no GPU)"
echo ""

mkdir -p outputs/logs outputs/checkpoints

echo "── Running EDL head fine-tune ─────────────────────"

python3 scripts/finetune_edl_head.py \
    --config configs/default.yaml \
    --config configs/bridges2.yaml \
    --checkpoint outputs/checkpoints/best_model.pt \
    --epochs 2

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done at $(date)"
echo ""
echo "  Check the log above for:"
echo "    ECE        — target < 0.15  (was 0.57)"
echo "    p_hat>0.3  — should be > 0% (was ~0%)"
echo "    p_hat>0.5  — should be > 0% (was 0%)"
echo ""
echo "  If ECE improved → run full retrain:"
echo "    sbatch scripts/slurm_train.sh"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
