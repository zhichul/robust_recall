#!/bin/bash
set -eu

if [ ! -f "install_dependencies.sh" ]; then
    echo "Please run this script in the root folder of the repo."
    exit 1
fi

# params
verl_commit=2c179dae234ca65b18ce8d2fe63d5b367910f628
proj_root=$(pwd)

# push some environment variables
echo "ROOT=$(pwd)" > .env

# create conda env
prefix=/local/zlu39/.conda_envs/robust_recall
mkdir -p ${prefix}
conda create --prefix ${prefix} python==3.10 -y
# ln -s ${prefix} ${HOME}/.conda/envs/local_robust_recall

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${prefix}
conda env list

################## get verl ####################
if [ -e "lib/verl" ]; then
    echo "Verl already downloaded, skipping clone."
else
    git clone https://github.com/volcengine/verl.git lib/verl
fi
cd lib/verl
git checkout $verl_commit
git apply ../patches/verl_torch_version.patch
pip3 install -e . --no-cache-dir

# Install the latest stable version of vLLM
pip3 install vllm==0.6.3 --no-cache-dir

# Install flash-attn
pip3 install flash-attn --no-build-isolation --no-cache-dir

# patch dapo threaded
git apply ../patches/verl_dapo_threaded.patch

cd $proj_root

pip3 freeze --exclude-editable > constraints.txt

################### get simple_evals ####################
if [ -e "lib/simple_evals" ]; then
    echo "simple_evals already downloaded, skipping clone."
else
    git clone https://github.com/openai/simple-evals.git lib/simple_evals
fi
if [ -e "lib/human-eval" ]; then
    echo "human-eval already downloaded, skipping clone."
else
    git clone https://github.com/openai/human-eval.git lib/human-eval
fi
cd lib
cd simple_evals
git checkout 3ec4e9b5ae3931a1858580e2fd3ce80c7fcbe1d9
git apply ../patches/simple_evals_chat.patch
cd ..
cd human-eval
git checkout 6d43fb980f9fee3c892a914eda09951f772ad10d
cd ..
set +eu
pip3 install -e human-eval -c $proj_root/constraints.txt --use-pep517
set -eu
pip3 install openai anthropic blobfile tabulate -c $proj_root/constraints.txt
cd $proj_root

################### get doc_cot ####################
if [ -e "lib/doc_cot" ]; then
    echo "doc_cot already downloaded, skipping clone."
else
    git clone https://github.com/zhichul/doc_cot.git lib/doc_cot
fi

cd lib/doc_cot
pip3 install -r requirements.txt -c $proj_root/constraints.txt
bash scripts/reassemble_bin.sh doc_cot/corpus/indices/olmo-mix-1124-pes2o-ids-to-file.parquet
echo "\
PES2O_PATH=/pscratch/sd/z/zlu39/olmo-mix-1124/data/pes2o/
S2_API_KEY=klTlPNR9qxaTKnP604LdT6TRzThTv21M9JFCI8h1
PROJECT_ROOT=$(pwd)
" > .env
cd $proj_root

################### requirements.txt ######################
pip install -r requirements.txt -c constraints.txt
pip install tensordict==0.6.2 -c constraints.txt

################### get data and model ####################
huggingface-cli download --repo-type dataset akariasai/PopQA
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct