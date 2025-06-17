#!/bin/bash
set -eu

# Default values
ngpus=4
model_path="allenai/OLMo-2-1124-7B-Instruct"
batch_size=8
lr=1e-6
max_length=3072
total_steps=
epochs=5
per_gpu_batch_size=1
eval_freq=50
grad_ckpt=True
lora_rank=0

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --ngpus) ngpus="$2"; shift 2 ;;
        --model_path) model_path="$2"; shift 2 ;;
        --batch_size) batch_size="$2"; shift 2 ;;
        --per_gpu_batch_size) per_gpu_batch_size="$2"; shift 2 ;;
        --lr) lr="$2"; shift 2 ;;
        --max_length) max_length="$2"; shift 2 ;;
        --total_steps) total_steps="$2"; shift 2 ;;
        --epochs) epochs="$2"; shift 2 ;;
        --graph_spec) graph_spec="$2"; shift 2 ;;
        --eval_freq) eval_freq="$2"; shift 2 ;;
        --grad_ckpt) grad_ckpt="$2"; shift 2 ;;
        --lora_rank) lora_rank="$2"; shift 2 ;;
        --data_seed) data_seed="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --ngpus <int> --model_path <path> --batch_size <int> --per_gpu_batch_size <int> --lr <float>"
            echo "  --max_length <int> "
            echo "  --total_steps <int> --epochs <int> --grad_ckpt <bool>"
            echo "  --graph_spec <str> --eval_freq <int> --lora_rank <int>"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done


project_name=rl4r
model_name=$(basename "$model_path" | sed 's/[^a-zA-Z0-9]/_/g')
experiment_name="exp_sft_model_${model_name}_bs${batch_size}_lr${lr}_len${max_length}_lora${lora_rank}"

# Output results
echo "ngpus: $ngpus"
echo "model_path: $model_path"
echo "batch_size: $batch_size"
echo "per_gpu_batch_size: $per_gpu_batch_size"
echo "lr: $lr"
echo "max_length: $max_length"
echo "total_steps: $total_steps"
echo "epochs: $epochs"
echo "eval_freq: $eval_freq"
echo "grad_ckpt: $grad_ckpt"
echo "lora_rank: $lora_rank"
echo "experiment_name: $experiment_name"


########################   MAIN SCRIPT #########################
export HF_TOKEN="${HF_TOKEN:-YOUR_HF_TOKEN}"
export HF_HOME="${HF_HOME:-YOUR_HF_HOME}"
# export WANDB_ENTITY=gluzi
export WANDB_API_KEY=8a464cf7b440ea91becb29da1874822e4f5273ed
export VLLM_ATTENTION_BACKEND=XFORMERS

set +eu
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /tmp/zlu39/.conda_envs/robust_recall
set -eu

# train_file= ,data/sft/split0/1000_to_10000/train.parquet,data/sft/split0/10000_to_100000/train.parquet,data/sft/split0/100000_to_inf/train.parquet]
# val_file= ,data/sft/split0/1000_to_10000/dev.parquet,data/sft/split0/10000_to_100000/dev.parquet,data/sft/split0/100000_to_inf/dev.parquet]
if [[ -z "${total_steps}" ]]; then
    train_step_cmd=""
else
    train_step_cmd="trainer.total_training_steps=${total_steps}"
fi

# use bf16 for model_dtype to avoid OOM when loading checkpoints
# TODO: watch for patches or patch this issue myself
export PYTHONPATH=${PYTHONPATH:-""}:$(realpath ../../lib/verl/)
export HYDRA_FULL_ERROR=1
torchrun --standalone --nnodes=1 --nproc_per_node=${ngpus} \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_batch_size=${batch_size} \
    data.train_files="['data/sft/split0/0_to_1000/train.parquet']" \
    data.val_files="['data/sft/split0/0_to_1000/dev.parquet']" \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    +data.prompt_dict_keys=['sft_prompt'] \
    +data.response_dict_keys=['sft_answer'] \
    data.micro_batch_size_per_gpu=${per_gpu_batch_size} \
    data.max_length=${max_length} \
    optim.lr=${lr} \
    model.partial_pretrain=${model_path} \
    model.enable_gradient_checkpointing=${grad_ckpt} \
    model.lora_rank=${lora_rank} \
    trainer.default_local_dir=checkpoints/${project_name}/${experiment_name}/ \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.total_epochs=${epochs} \
    ${train_step_cmd} \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null \
    +trainer.eval_steps=${eval_freq} \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=True \
    > logs/${experiment_name}.out 2> logs/${experiment_name}.err