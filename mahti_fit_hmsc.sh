#!/bin/bash
#SBATCH --job-name=geolife_hmsc
#SBATCH --account=project_1234567
#SBATCH --output=output/%A
#SBATCH --ntasks=1 --cpus-per-task=32
#SBATCH --time=03:15:00 --partition=gpusmall
#SBATCH --gres=gpu:a100:1


IND=$SLURM_ARRAY_TASK_ID
NC=${1:-69}
NS=2519
NP=${2:-100}
NF=${3:-10}
SAM=${4:-100}
THIN=${5:-10}

mkdir -p output
module load tensorflow/2.18
hostname

data_path="${MAHTI_DATA_PATH}"
if [ -z "$data_path" ]; then echo "MAHTI_DATA_PATH not set"; exit 1; fi

input_path=$data_path/init/$(printf "init_nc%.4d_ns%.4d_np%.4d_nf%.2d_chain01.rds" $NC $NS $NP $NF)
output_path=$data_path/fmTF_mahti/$(printf "TF_nc%.4d_ns%.4d_np%.4d_nf%.2d_chain01_sam%.4d_thin%.4d.rds" $NC $NS $NP $NF $SAM $THIN)


srun python3 -m hmsc.run_gibbs_sampler --input $input_path --output $output_path --samples $SAM --transient $(($SAM*$THIN)) --thin $THIN --verbose 100 --fse 1
 
