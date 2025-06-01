import json
import random
import re
import os
import os.path
from datasets import load_dataset, concatenate_datasets, Dataset

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
    
    args = parser.parse_args()
    if args.decontaminated:
        dev_name = 'dev.decon'
        test_name = 'test.decon'
    else:
        dev_name = 'dev'
        test_name = 'test'
    train_datasets = []
    dev_datasets = []
    test_datasets = []
    train_datasets.append(load_dataset('parquet', data_files=os.path.join(args.data_source, f'train.parquet'))['train'])
    dev_datasets.append(load_dataset('parquet', data_files=os.path.join(args.data_source, f'{dev_name}.parquet'))['train'])
    test_datasets.append(load_dataset('parquet', data_files=os.path.join(args.data_source, f'{test_name}.parquet'))['train'])
        
    train_dataset = concatenate_datasets(train_datasets)
    dev_dataset = concatenate_datasets(dev_datasets)
    test_dataset = concatenate_datasets(test_datasets)
    def process_fn(example):
        user_prompt = example.pop('question')
        answer = example.pop('possible_answers')
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
                "answer": answer
            }
        }
        return data

    train_dataset = train_dataset.map(function=process_fn)
    dev_dataset = dev_dataset.map(function=process_fn)
    test_dataset = test_dataset.map(function=process_fn)

    local_dir = args.local_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    dev_dataset.to_parquet(os.path.join(local_dir, f'{dev_name}.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, f'{test_name}.parquet'))