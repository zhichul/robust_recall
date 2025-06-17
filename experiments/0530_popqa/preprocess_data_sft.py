import json
import random
import re
import os
import os.path
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer
import numpy as np
import argparse

import re
import numpy as np


if __name__ == '__main__':
    import random
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default=None)
    parser.add_argument('--data_source', default=None)
    parser.add_argument('--data_name', type=str, default=None)
    parser.add_argument('--decontaminated', action='store_true', default=False)
    parser.add_argument('--tokenizer', default='allenai/OLMo-2-1124-7B-Instruct')
    
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    doc_token_limit = 2048
    if args.decontaminated:
        dev_name = 'dev.decon'
        test_name = 'test.decon'
    else:
        dev_name = 'dev'
        test_name = 'test'
    train_datasets = []
    dev_datasets = []
    test_datasets = []
    train_datasets.append(load_dataset('parquet', data_files=os.path.join(args.data_source, f'train.wikilinked.parquet'))['train'])
    dev_datasets.append(load_dataset('parquet', data_files=os.path.join(args.data_source, f'{dev_name}.wikilinked.parquet'))['train'])
    test_datasets.append(load_dataset('parquet', data_files=os.path.join(args.data_source, f'{test_name}.wikilinked.parquet'))['train'])
        
    train_dataset = concatenate_datasets(train_datasets)
    dev_dataset = concatenate_datasets(dev_datasets)
    test_dataset = concatenate_datasets(test_datasets)
    dev1_dataset = Dataset.from_dict(test_dataset[:100])  # for monitoring with neither wiki nor qa data
    test_dataset = Dataset.from_dict(test_dataset[100:])

    def filter_wikilink(example):
        return len(example['s_docs']) > 0

    def process_fn_wiki(example):
        user_prompt = example.pop('question')
        answer = example.pop('possible_answers')
        subj = example.pop('subj')
        article = tokenizer.decode(tokenizer(example['s_docs'][0]['text'])['input_ids'][:2048])
        data = {
            "data_source": (args.data_source if args.data_name is None else args.data_name),
            "prompt": [
                {
                "role": "system",
                "content": "You are a helpful assistant. When responding to questions, enclose your thoughts in <thought></thought> and your clean, final answer in <answer></answer>."
            },
            {
                "role": "user",
                "content": user_prompt
            }],
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                "index": example.pop('id'),
                "metadata": example,
                "problem": user_prompt,
                "answer": answer,
                "sft_prompt": f"Try to recollect the wikipedia article on {subj}.",
                "sft_answer": f"{article}...",
            
            }
        }
        return data

    def process_fn_qa(example):
        user_prompt = example.pop('question')
        answer = example.pop('possible_answers')
        subj = example.pop('subj')
        article = tokenizer.decode(tokenizer(example['s_docs'][0]['text'])['input_ids'][:128])
        data = {
            "data_source": (args.data_source if args.data_name is None else args.data_name),
            "prompt": [
                {
                "role": "system",
                "content": "You are a helpful assistant. When responding to questions, enclose your thoughts in <thought></thought> and your clean, final answer in <answer></answer>."
            },
            {
                "role": "user",
                "content": user_prompt
            }],
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                "index": example.pop('id'),
                "metadata": example,
                "problem": user_prompt,
                "answer": answer,
                "sft_prompt": f"You are a helpful assistant. When responding to questions, enclose your thoughts in <thought></thought> and your clean, final answer in <answer></answer>. \n{user_prompt}",
                "sft_answer": f"<thought> Here's what I remember of the wikipedia article on {subj}: {article}... </thought> <answer>{json.loads(answer)[0]}</answer>",
            }
        }
        return data
    train_dataset = train_dataset.filter(filter_wikilink)
    dev_dataset = dev_dataset.filter(filter_wikilink)
    dev1_dataset = dev1_dataset.filter(filter_wikilink)

    train_qa = train_dataset.map(function=process_fn_qa)
    train_wiki = train_dataset.map(function=process_fn_wiki)
    dev_wiki = dev_dataset.map(function=process_fn_wiki)

    dev1_qa = dev1_dataset.map(function=process_fn_qa)
    dev1_wiki = dev1_dataset.map(function=process_fn_wiki)

    train_dataset = concatenate_datasets([train_qa, train_wiki, dev_wiki])
    dev_dataset = concatenate_datasets([dev1_qa, dev1_wiki])

    local_dir = args.local_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    dev_dataset.to_parquet(os.path.join(local_dir, f'dev.parquet'))
