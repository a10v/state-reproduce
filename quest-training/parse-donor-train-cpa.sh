#!/bin/bash
#SBATCH --account=p32775
#SBATCH --partition=gengpu              # GPU partition (48 h max)
#SBATCH --gres=gpu:a100:1               # lightning trainer uses a single GPU
#SBATCH --nodes=1
#SBATCH --mem=150G                      # Parse dataset is ~212 GB; leave headroom
#SBATCH --time=47:59:00                 # <= 48h limit on gengpu
#SBATCH --job-name=cpa-parse-donor
#SBATCH --output=cpa_parse_donor_%j.out
#SBATCH --error=cpa_parse_donor_%j.err

# --- Environment (cluster-standard) ---
module load python-miniconda3
eval "$(conda shell.bash hook)"
source activate /projects/p32775/pythonenvs/state_sets   # change to your env

module load gcc/12.4.0-gcc-8.5.0
module load cuda/12.4.0-gcc-12.4.0
module load git

# --- Basic diagnostics ---
mkdir -p logs save
echo "SLURM_JOBID=${SLURM_JOBID}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION}"
echo "SLURM_NNODES=${SLURM_NNODES}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE}"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}"
export NCCL_IB_DISABLE=1
which python
python -V || true
nvcc --version || true
python -c "import torch; print('CUDA:', torch.version.cuda, 'GPUs:', torch.cuda.device_count())" || true

export WANDB_CACHE_DIR=/projects/p32775/.cache
export WANDB_MODE=offline                 # disable WANDB network writes
export HF_DATASETS_CACHE=/projects/p32775/.cache/hf_datasets

# --- Job params ---
PROJECT_ROOT=/projects/p32775/state-reproduce/baselines          # repo checkout
DATA_TOML=/projects/p32775/state_toml_files/parse_tomls/donor.toml  # the file you showed
OUTPUT_DIR=/projects/p32775/state_runs/cpa_parse_donor          # large shared storage
mkdir -p "${OUTPUT_DIR}"

cd "${PROJECT_ROOT}"

# --- Launch (single-GPU lightning trainer) ---
srun python -m state_sets_reproduce.train \
    data.kwargs.toml_config_path=${DATA_TOML} \
    data.kwargs.embed_key=X_hvg \
    data.kwargs.basal_mapping_strategy=random \
    data.kwargs.output_space=gene \
    data.kwargs.num_workers=24 \
    data.kwargs.batch_col=cell_type \
    data.kwargs.pert_col=cytokine \
    data.kwargs.cell_type_key=donor \
    data.kwargs.control_pert=PBS \
    data.kwargs.map_controls=true \
    model=cpa \
    training=cpa \
    training.max_steps=250000 \
    training.val_freq=5000 \
    training.test_freq=9000 \
    training.batch_size=128 \
    wandb.tags="[cpa,parse,donor]" \
    use_wandb=false \
    output_dir=${OUTPUT_DIR} \
    name=donor \
    overwrite=false \
    hydra.run.dir=. \
    hydra.output_subdir=null | awk '{ print strftime("[%Y-%m-%d %H:%M:%S]"), $0; fflush(); }'
