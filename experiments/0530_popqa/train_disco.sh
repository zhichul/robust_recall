#!/usr/bin/env bash
set -eu

# Default values
split=split0
subsplits=()
ngpus=4
model_path="meta-llama/Llama-3.1-8B-Instruct"
batch_size=$((ngpus * 2))
lr=1e-6
n_rollout=32
max_prompt_length=2048
max_response_length=4096
total_steps=10000
epochs=1000
vllm_gpu_util=0.75
per_gpu_batch_size=1
tensor_parallel=2
max_checkpoints=2
eval_freq=50
save_freq=10
actor_offload=False
resume_mode=auto
log_val_n=10
reward_fn=score_judge
entropy_coeff=0
rollout_temp=1.0
max_num_gen_batches=30
reward_manager=dapo_threaded

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --ngpus) ngpus="$2"; shift 2 ;;
        --model_path) model_path="$2"; shift 2 ;;
        --batch_size) batch_size="$2"; shift 2 ;;
        --per_gpu_batch_size) per_gpu_batch_size="$2"; shift 2 ;;
        --lr) lr="$2"; shift 2 ;;
        --n_rollout) n_rollout="$2"; shift 2 ;;
        --max_prompt_length) max_prompt_length="$2"; shift 2 ;;
        --max_response_length) max_response_length="$2"; shift 2 ;;
        --total_steps) total_steps="$2"; shift 2 ;;
        --epochs) epochs="$2"; shift 2 ;;
        --vllm_gpu_util) vllm_gpu_util="$2"; shift 2 ;;
        --tensor_parallel) tensor_parallel="$2"; shift 2 ;;
        --max_checkpoints) max_checkpoints="$2"; shift 2 ;;
        --split) split="$2"; shift 2 ;;
        --subsplit) subsplits+=("$2"); shift 2 ;;
        --save_freq) save_freq="$2"; shift 2 ;;
        --eval_freq) eval_freq="$2"; shift 2 ;;
        --actor_offload) actor_offload="$2"; shift 2 ;;
        --resume_mode) resume_mode="$2"; shift 2 ;;
        --log_val_n) log_val_n="$2"; shift 2 ;;
        --reward_fn) reward_fn="$2"; shift 2 ;;
        --entropy_coeff) entropy_coeff="$2"; shift 2 ;;
        --rollout_temp) rollout_temp="$2"; shift 2 ;;
        --max_num_gen_batches) max_num_gen_batches="$2"; shift 2 ;;
        --reward_manager) reward_manager="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --ngpus <int> --model_path <path> --batch_size <int> --per_gpu_batch_size <int> --lr <float>"
            echo "  --n_rollout <int> --max_prompt_length <int> --max_response_length <int>"
            echo "  --total_steps <int> --epochs <int> --vllm_gpu_util <float> --tensor_parallel <int>"
            echo "  --max_checkpoints <int> --split <str> --save_freq <int> --eval_freq <int>"
            echo "  --actor_offload <bool> --resume_mode <auto|disable> --log_val_n <int> --reward_fn <str> --max_num_gen_batches <int> --reward_manager <str> --subsplit <str>"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ ${#subsplits[@]} -eq 0 ]]; then
    # default buckets when none are supplied
    subsplits=(0_to_1000 1000_to_10000 10000_to_100000 100000_to_inf)
fi

train_files=()
for r in "${subsplits[@]}"; do
    train_files+=("'data/processed/${split}/${r}/train.parquet'")
done

split=split0
project_name=rl4r
model_name=$(basename "$model_path" | sed 's/[^a-zA-Z0-9]/_/g')
subsplits_str="$(IFS='-'; echo "${subsplits[*]}")"
experiment_name="popqa_exp_dapo_model_${model_name}_bs${batch_size}_lr${lr}_roll${n_rollout}_p${max_prompt_length}_r${max_response_length}_rwd${reward_fn}_ent${entropy_coeff}_rt${rollout_temp}_d${split}_s${subsplits_str}"

# Output results
echo "ngpus: $ngpus"
echo "model_path: $model_path"
echo "batch_size: $batch_size"
echo "per_gpu_batch_size: $per_gpu_batch_size"
echo "lr: $lr"
echo "n_rollout: $n_rollout"
echo "max_prompt_length: $max_prompt_length"
echo "max_response_length: $max_response_length"
echo "total_steps: $total_steps"
echo "epochs: $epochs"
echo "vllm_gpu_util: $vllm_gpu_util"
echo "tensor_parallel: $tensor_parallel"
echo "max_checkpoints: $max_checkpoints"
echo "split: $split"
echo "subsplits: ${subsplits[*]}"
echo "save_freq: $save_freq"
echo "eval_freq: $eval_freq"
echo "actor_offload: $actor_offload"
echo "resume_mode: $resume_mode"
echo "log_val_n: $log_val_n"
echo "reward_fn: $reward_fn"
echo "entropy_coeff: $entropy_coeff"
echo "rollout_temp: $rollout_temp"
echo "max_num_gen_batches: $max_num_gen_batches"
echo "reward_manager: $reward_manager"
echo "experiment_name: $experiment_name"

############
# export HF_TOKEN="${HF_TOKEN:-YOUR_HF_TOKEN}"
# export HF_HOME="${HF_HOME:-YOUR_HF_HOME}"
export WANDB_API_KEY=8a464cf7b440ea91becb29da1874822e4f5273ed
# export VLLM_ATTENTION_BACKEND=XFORMERS

set +eu
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /local/zlu39/.conda_envs/robust_recall/
set -eu

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$max_prompt_length
max_response_length=$max_response_length
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=$max_num_gen_batches
train_prompt_bsz=$batch_size
gen_prompt_bsz=$((train_prompt_bsz))
n_resp_per_prompt=$n_rollout
train_prompt_mini_bsz=$ngpus

# Ray
# RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
# WORKING_DIR=${WORKING_DIR:-"${PWD}"}
# RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
# NNODES=${NNODES:-16}
NNODES=1

# Paths
# RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
# MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen2.5-32B"}
# CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
# TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
# TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}

# Algorithm
temperature=$rollout_temp
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))

TRAIN_FILE="[$(IFS=', '; echo "${train_files[*]}")]"
TEST_FILE="['data/processed/${split}/0_to_1000/dev.decon.parquet', 'data/processed/${split}/1000_to_10000/dev.decon.parquet', 'data/processed/${split}/10000_to_100000/dev.decon.parquet', 'data/processed/${split}/100000_to_inf/dev.decon.parquet']"
set +eu
export PYTHONPATH=$PYTHONPATH:$(realpath ../../lib/verl)
set -eu
export HYDRA_FULL_ERROR=1
python3 -u -m recipe.dapo.main_dapo \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${ngpus} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${model_path}" \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_offload} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${vllm_gpu_util} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tensor_parallel} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${actor_offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=${reward_manager} \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.n_gpus_per_node=${ngpus} \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=${eval_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${epochs} \
    trainer.resume_from_path=checkpoints/${project_name}/${experiment_name} \
    trainer.resume_mode=${resume_mode} \
    custom_reward_function.path=${reward_fn}.py \
    trainer.log_val_generations=${log_val_n} \
    hydra.searchpath="['file://$(realpath ../../lib/verl)/verl/trainer/config']" \
    > logs/${experiment_name}.out 2> logs/${experiment_name}.err