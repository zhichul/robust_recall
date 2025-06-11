from concurrent.futures import ThreadPoolExecutor
import dataclasses
import json
import os
import numpy as np
from openai import OpenAI
from rouge_score import rouge_scorer
import re
import argparse

import tqdm

from datasets import load_dataset
from memorization_prompt import answer_prefix_extraction_prompt, subject_prefix_extraction_prompt

parser = argparse.ArgumentParser()
parser.add_argument('--base_model_host', default='127.0.0.1')
parser.add_argument('--instruct_model_host', default='127.0.0.1')
parser.add_argument('--llm_host', default='127.0.0.1')
parser.add_argument('--base_model_name', default='allenai/OLMo-2-1124-7B')
parser.add_argument('--instruct_model_name', default='allenai/OLMo-2-1124-7B-Instruct')
parser.add_argument('--llm_name', default='meta-llama/Llama-3.3-70B-Instruct')
parser.add_argument('--fold', default=0, type=int)
parser.add_argument('--subset', default=str, type='dev')
parser.add_argument('--n', default=int, type=64)
cmdline_args = parser.parse_args()

client_base = OpenAI(base_url=f'http://{cmdline_args.base_model_host}:8001/v1', api_key="EMPTY")
client_ins = OpenAI(base_url=f'http://{cmdline_args.instruct_model_host}:8001/v1', api_key="EMPTY")
client_llm = OpenAI(base_url=f'http://{cmdline_args.llm_host}:8001/v1', api_key="EMPTY")
model_base = cmdline_args.base_model_name
model_ins = cmdline_args.instruct_model_name
model_llm = cmdline_args.llm_name
import transformers
transformers.__version__
from transformers import AutoTokenizer

tokenizer_ins = AutoTokenizer.from_pretrained(cmdline_args.instruct_model_name)
tokenizer_base = AutoTokenizer.from_pretrained(cmdline_args.base_model_name)

def get_completions(prefix, model=model_base, temperature=1.0, max_completion_tokens=32, n=1):
    output = client_base.completions.create(model=model, prompt=prefix, max_tokens= max_completion_tokens, temperature=temperature, n=n)
    return [c.text for c in output.choices]

prompt_simple = "You are a helpful assistant. When responding to questions, enclose your thoughts in <thought></thought> and your clean, final answer in <answer></answer>. The answer should be a short phrase or name rather than a sentence."
prompt_gen_read = prompt_simple + " When you are asked a factual question about an entity, you must try to recollect by yourself the wikipedia article about the entity first without using any external tools."

def get_answers(question, model=model_ins, temperature=1.0, max_completion_tokens=1024, gen_read=False, n=1):
    output = client_ins.chat.completions.create(model=model, messages=[{
                "role": "system",
                "content": prompt_simple if not gen_read else prompt_gen_read
            }, {'role': 'user', 'content': question}], max_completion_tokens=max_completion_tokens, temperature=temperature, n=n)
    return [c.message.content for c in output.choices]

def call_llm(messages, model=model_llm, temperature=0, max_completion_tokens=1024, n=1):
    output = client_ins.chat.completions.create(model=model, messages=messages, max_completion_tokens=max_completion_tokens, temperature=temperature, n=n)
    return [c.message.content for c in output.choices]

def extract_answer(solution_str):
    return re.findall(r'<answer>(.*?)</answer>', solution_str, re.IGNORECASE)[0]

default_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

@dataclasses.dataclass
class ScoreAtK:
    mean: float
    sem: float
    k: int
    mc_samples_outer: int
    mc_samples_inner: int
    sample_scores: list[float] # len = mc_samples
    raw_scores: list[float] # len = size of answer pool

def bootstrap_no_replacement(data, k, n, seed=None):
    rng = np.random.default_rng(seed)
    data = np.asarray(data)

    # Collect mean of each resample
    means = np.array([rng.choice(data, size=k, replace=False).mean()
                      for _ in range(n)])

    mean_hat = means.mean()
    sem_hat = means.std(ddof=1)          # bootstrap SEM

    return mean_hat, sem_hat

def score_at_k(label: str|list[str], answers: list[str], scoring_fn, k=1, mc_outer_resample=None, mc_inner_resample=None, seed=None):
    scores = []
    for answer in answers:
        scores.append(scoring_fn(label, answer))
    if k == 1 and mc_outer_resample is None:
        return ScoreAtK(
            np.mean(scores),
            np.std(scores, ddof=1) / np.sqrt(len(scores)),
            k,
            None,
            None,
            None,
            scores,
        )
    else:
        rng = np.random.default_rng(seed)
        scores_np = np.asarray(scores)

        # collect max of each resample
        bootstrap_outer = rng.choice(scores_np, size=(mc_outer_resample, len(scores)), replace=True)
        row_index = np.broadcast_to(np.arange(mc_outer_resample)[:,None,None], (mc_outer_resample, mc_inner_resample, k))
        choice_idx = rng.choice(np.arange(len(scores)),size=(mc_outer_resample, mc_inner_resample, k), replace=True)
        bootstrap_inner = bootstrap_outer[row_index, choice_idx]
        assert bootstrap_inner.shape == (mc_outer_resample, mc_inner_resample, k)
        best_at_k_sampled_scores = bootstrap_inner.max(-1)
        best_at_k_means = best_at_k_sampled_scores.mean(-1)
        best_at_k_vars = best_at_k_sampled_scores.var(-1, ddof=1)
        var_mean = best_at_k_means.var(ddof=1) -  best_at_k_vars.mean(-1) / mc_inner_resample
        sem = np.sqrt(var_mean)
        mean = best_at_k_means.mean()
        return ScoreAtK(
            mean,
            sem,
            k,
            mc_outer_resample,
            mc_inner_resample,
            best_at_k_sampled_scores,
            scores
        )
    
def main_answer():
    out_dir = f'generations/memo/fifty_fifty/split{cmdline_args.fold}/{freq}'
    os.makedirs(out_dir, exist_ok=True)
    for split in [cmdline_args.fold]:
        for freq in tqdm(['0_to_1000', '1000_to_10000', '10000_to_100000', '100000_to_inf']):
            dataset = load_dataset('parquet', data_files=f"data/raw/splits/fifty_fifty/split{split}/{freq}/{cmdline_args.subset}.wikilinked.parquet")['train']
            answers = []
            with ThreadPoolExecutor(max_workers=32) as opool:
                for i in tqdm(range(len(dataset))):
                    instance = dataset[i]
                    answers.append(opool.submit(get_answers, instance['question'], n=cmdline_args.n))
            answers = json.dumps([ans.result() for ans in answers])
            dataset = dataset.add_column('model_predictions', answers)
            dataset.to_parquet(f'{out_dir}/{cmdline_args.subset}.wikilinked.qa.parquet')

def prefix_len_up_to_subject(wiki, instance):
    user = f"""\
SUBJECT: {[instance['subj']] + json.loads(instance['s_aliases'])}
RELATION: {instance['prop']}
OBJECT: {[instance['obj']] + json.loads(instance['o_aliases'])}
QUESTION: {instance['question']}
ANSWERS: {instance['possible_answers']}
ARTICLE:```{wiki}
```
"""
    response = call_llm(messages=[
        {'role': 'system', 'content': subject_prefix_extraction_prompt},
        {'role': 'user', 'content': user}
    ])[0]
    return response


def prefix_up_to_answer(wiki, instance):
    user = f"""\
SUBJECT: {[instance['subj']] + json.loads(instance['s_aliases'])}
RELATION: {instance['prop']}
OBJECT: {[instance['obj']] + json.loads(instance['o_aliases'])}
QUESTION: {instance['question']}
ANSWERS: {instance['possible_answers']}
ARTICLE:```{wiki}
```
"""
    response = call_llm(messages=[
        {'role': 'system', 'content': answer_prefix_extraction_prompt},
        {'role': 'user', 'content': user}
    ])[0]
    return response

def main_completion():
    out_dir = f'generations/memo/fifty_fifty/split{cmdline_args.fold}/{freq}'
    os.makedirs(out_dir, exist_ok=True)
    for split in [cmdline_args.fold]:
        for freq in tqdm(['0_to_1000', '1000_to_10000', '10000_to_100000', '100000_to_inf']):
            dataset = load_dataset('parquet', data_files=f"data/raw/splits/fifty_fifty/split{split}/{freq}/{cmdline_args.subset}.wikilinked.parquet")['train']
            answers = []
            with ThreadPoolExecutor(max_workers=32) as opool:
                for i in tqdm(range(len(dataset))):
                    instance = dataset[i]
                    answers.append(opool.submit(get_completions, instance['question'], n=cmdline_args.n))
            answers = json.dumps([ans.result() for ans in answers])
            dataset = dataset.add_column('model_predictions', answers)
            dataset.to_parquet(f'{out_dir}/{cmdline_args.subset}.wikilinked.completions.parquet')



if __name__ == "__main__":
    s1 =score_at_k('hello', ['hello world', 'world', 'hello', 'hello world world'] * 100, lambda x, y: default_scorer.score(x, y)['rougeL'].fmeasure, k=1)
    s2 =score_at_k('hello', ['hello world', 'world', 'hello', 'hello world world'] * 100, lambda x, y: default_scorer.score(x, y)['rougeL'].fmeasure, k=1, mc_outer_resample=100000, mc_inner_resample=100)
    s3 =score_at_k('hello', ['hello world', 'world', 'hello', 'hello world world'] * 100, lambda x, y: default_scorer.score(x, y)['rougeL'].fmeasure, k=1, mc_outer_resample=100000, mc_inner_resample=100)
    print(s1.mean, s1.sem)
    print(s2.mean, s2.sem)
    print(s3.mean, s3.sem)
