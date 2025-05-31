import json
import random
import re
import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "lib")))
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__))))

NODES = None
EVALUATORS = None

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if isinstance(ground_truth, list):
        ground_truth = json.dumps(ground_truth)
        print(ground_truth)
    if EVALUATORS is None:
        init_nodes()
    problem = extra_info['problem']
    evaluator = random.choice(EVALUATORS)
    solution_m = re.search(r'<answer>(.*?)</answer>', solution_str, re.IGNORECASE)
    if solution_m is None:
        score = 0.0
        extracted = "format_error"
        judgement = "format_error"
    else:
        extracted = solution_m.group(1)
        if extracted == "":
            extracted = "empty_answer"
        judgement = evaluator.grade_sample(problem, ground_truth, extracted)
        if judgement == "A":
            score = 1.0
        else:
            score = 0.1
    return {'score': score, 'acc': float(score == 1.0), 'pred': solution_str, 'judge_pred': judgement, 'extracted_answer': extracted}

def init_nodes():
    from sq_utils import running_nodes
    from popqa_eval import PopQAEval
    from simple_evals.sampler.claude_sampler import ClaudeCompletionSampler, CLAUDE_SYSTEM_MESSAGE_LMSYS
    from simple_evals.sampler.o_chat_completion_sampler import OChatCompletionSampler
    from simple_evals.sampler.responses_sampler import ResponsesSampler
    from simple_evals.sampler.chat_completion_sampler import (
        OPENAI_SYSTEM_MESSAGE_API,
        OPENAI_SYSTEM_MESSAGE_CHATGPT,
        ChatCompletionSampler,
    )
    global NODES
    global EVALUATORS
    NODES = running_nodes('vllm')
    EVALUATORS = []
    for node in NODES: # technically, each node is a nodelist, but we have singletons by assumption
        grading_sampler = ChatCompletionSampler(
                model="meta-llama/Llama-3.3-70B-Instruct",
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                max_tokens=2048,
                base_url=f'http://{node}:8001/v1',
                api_key='EMPTY'
            )
        evaluator = PopQAEval(
            grader_model=grading_sampler,
            num_examples=None,
        )
        EVALUATORS.append(evaluator)

if __name__ == "__main__":
    test_problem = 'On what day, month, and year did Tara Chand (a politician and a Dalit leader from Jammu and Kashmir) resign from the Indian National Congress in support of Ghulam Nabi Azad?'
    test_answer = '<thought> some random thoughts </thought> <answer> Tara Chand, a politician and Dalit leader from Jammu and Kashmir, resigned from the Indian National Congress on **30th August 2022** in support of Ghulam Nabi Azad. </answer>'
    test_label = 'August 30, 2022'
    print(compute_score(None, test_answer, [test_label, test_label], {'problem': test_problem}))
    print(compute_score(None, test_answer, test_label, {'problem': test_problem}))