import json
import random
import re

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    problem = extra_info['problem']
    solution_m = re.search(r'<answer>(.*?)</answer>', solution_str, re.IGNORECASE)
    if solution_m is None:
        score = 0.0
        extracted = "format_error"
        judgement = "format_error"
    else:
        extracted = solution_m.group(1)
        if extracted == "":
            extracted = "empty_answer"
            judgement = "not_attempted"
        else:
            judgement = "incorrect"
            for possible_answer in ground_truth:
                if extracted.strip().lower().startswith(possible_answer.lower()):
                    judgement = "correct"
        if judgement == "correct":
            score = 1.0
        else:
            score = 0.1
    return {'score': score, 'acc': float(score == 1.0), 'pred': solution_str, 'judge_pred': judgement, 'extracted_answer': extracted}

if __name__ == "__main__":
    test_problem = 'On what day, month, and year did Tara Chand (a politician and a Dalit leader from Jammu and Kashmir) resign from the Indian National Congress in support of Ghulam Nabi Azad?'
    test_answer = '<thought> some random thoughts </thought> <answer> August 30, 2022 </answer>'
    test_label = ['August 30 2022', 'august 30, 2022']
    print(compute_score(None, test_answer, test_label, {'problem': test_problem}))
    print(compute_score(None, test_answer, test_label, {'problem': test_problem}))