#!/bin/bash
#SBATCH --job-name=fluidum-sim
#SBATCH --partition=long         # <--- Change your partition here
#SBATCH --time=48:00:00            # <--- Change max time here (HH:MM:SS)
#SBATCH --cpus-per-task=48          # <--- Change threads here
#SBATCH --mem=32G                  # <--- Change memory here
#SBATCH --output=logs/sim_%A_%a.out
#SBATCH --error=logs/sim_%A_%a.err

# 1. Capture Arguments
JULIA_SCRIPT="$1"
CONFIG_YAML="$2"

# 2. Validation
if [ -z "$JULIA_SCRIPT" ] || [ -z "$CONFIG_YAML" ]; then
    echo "❌ Error: Missing arguments."
    echo "Usage: sbatch run_simulation.sh <script.jl> <config.yaml>"
    exit 1
fi

mkdir -p logs

# 3. Environment Setup
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JULIA_PKG_PRECOMPILE_AUTO=0

# Define your image path (hardcoded for simplicity)
CONTAINER_IMG="/lustre/alice/users/fcapell/julia_apptainer.sif"

echo "🚀 Job started on $(hostname) at $(date)"
echo "🧵 Threads: $JULIA_NUM_THREADS"

# 4. Execute via Apptainer
# We bind the current directory so Julia can see both the .jl and the .yaml
apptainer exec --bind $PWD:/mnt --pwd /mnt "$CONTAINER_IMG" \
    julia --project=. "$JULIA_SCRIPT" "$CONFIG_YAML"

echo "✅ Job finished at $(date)"