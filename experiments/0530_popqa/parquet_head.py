import sys

input = sys.argv[1]
lines = int(sys.argv[2])
output = sys.argv[3]

from datasets import load_dataset, Dataset

d = load_dataset('parquet', data_files=input)['train']
Dataset.from_dict(d[:lines]).to_parquet(output)