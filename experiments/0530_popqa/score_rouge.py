import json
import random
import re
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def compute_score(data_source, solution_str, ground_truth, extra_info=None, multiline=False):
    ground_truth = json.loads(ground_truth) # for some reason popqa ground truth is a json string
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    flag = re.IGNORECASE
    if multiline:
        flag = flag | re.DOTALL
    solutions = re.findall(r'<answer>(.*?)</answer>', solution_str, flag)
    if len(solutions) != 1:
        score = 0.0
        extracted = "format_error"
        rl = 0.0
    else:
        extracted = solutions[0]
        score = 0.1
        rl = scorer.score_multi(ground_truth, extracted)['rougeL'].fmeasure
        score += 0.9 * rl
    return {'score': score, 'acc': rl, 'pred': solution_str, 'extracted_answer': extracted}

if __name__ == "__main__":
    test_problem = 'On what day, month, and year did Tara Chand (a politician and a Dalit leader from Jammu and Kashmir) resign from the Indian National Congress in support of Ghulam Nabi Azad?'
    test_answer = '<thought> some random thoughts </thought> <answer> August 30, 2022 </answer>'
    test_label = ['August 30 2022', 'august 30, 2022']
    print(compute_score(None, test_answer, test_label, {'problem': test_problem}))
    print(compute_score(None, test_answer, test_label, {'problem': test_problem}))