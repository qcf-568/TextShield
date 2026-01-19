import os
import re
from typing import Dict, List, Union

import json

from swift.llm import InferRequest


class ORM:

    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class ReactORM(ORM):

    @staticmethod
    def evaluate_action_reward(action_pred: list, action_ref: list, cand_list: list, ref_list: list):
        f1 = []
        for i in range(len(action_pred)):
            ref_action = action_ref[i]
            pred_action = action_pred[i]

            ref_input = ref_list[i]
            cand_input = cand_list[i]

            ref_is_json = False
            try:
                ref_input_json = json.loads(ref_input)
                ref_is_json = True
            except Exception:
                ref_input_json = ref_input

            cand_is_json = False
            try:
                cand_input_json = json.loads(cand_input)
                cand_is_json = True
            except Exception:
                cand_input_json = cand_input

            if ref_action != pred_action or (ref_is_json ^ cand_is_json):
                f1.append(0)
            elif not ref_is_json and not cand_is_json:
                rougel = ReactORM.evaluate_rougel([ref_input_json], [cand_input_json])
                if rougel is None or rougel < 10:
                    f1.append(0)
                elif 10 <= rougel < 20:
                    f1.append(0.1)
                else:
                    f1.append(1)
            else:
                if not isinstance(ref_input_json, dict) or not isinstance(cand_input_json, dict):
                    # This cannot be happen, but:
                    # line 62, in evaluate_action_reward
                    # for k, v in ref_input_json.items():
                    # AttributeError: 'str' object has no attribute 'items'
                    # print(f'>>>>>>ref_input_json: {ref_input_json}, cand_input_json: {cand_input_json}')
                    f1.append(0)
                    continue

                half_match = 0
                full_match = 0
                if ref_input_json == {}:
                    if cand_input_json == {}:
                        f1.append(1)
                    else:
                        f1.append(0)
                else:
                    for k, v in ref_input_json.items():
                        if k in cand_input_json.keys():
                            if cand_input_json[k] == v:
                                full_match += 1
                            else:
                                half_match += 1

                    recall = (0.5 * half_match + full_match) / (len(ref_input_json) + 1e-30)
                    precision = (0.5 * half_match + full_match) / (len(cand_input_json) + 1e-30)
                    try:
                        f1.append((2 * recall * precision) / (recall + precision))
                    except Exception:
                        f1.append(0.0)

        if f1[0] == 1.0:
            return True
        else:
            return False

    @staticmethod
    def parse_action(text):
        if 'Action Input:' in text:
            input_idx = text.rindex('Action Input:')
            action_input = text[input_idx + len('Action Input:'):].strip()
        else:
            action_input = '{}'

        if 'Action:' in text:
            action_idx = text.rindex('Action:')
            action = text[action_idx + len('Action:'):].strip()
            if 'Action Input:' in action:
                input_idx = action.index('Action Input:')
                action = action[:input_idx].strip()
        else:
            action = 'none'
        return action, action_input

    @staticmethod
    def parse_output(text):
        action, action_input = ReactORM.parse_action(text)
        return action, action_input

    def __call__(self, infer_requests: List[Union[InferRequest, Dict]], solution: List[str], **kwargs) -> List[float]:
        rewards = []
        if not isinstance(infer_requests[0], str):
            predictions = [request['messages'][-1]['content'] for request in infer_requests]
        else:
            predictions = infer_requests
        for prediction, ground_truth in zip(predictions, solution):
            if prediction.endswith('Observation:'):
                prediction = prediction[:prediction.index('Observation:')].strip()
            action_ref = []
            action_input_ref = []
            action_pred = []
            action_input_pred = []
            reference = ground_truth
            prediction = prediction.replace('<|endoftext|>', '').replace('<|im_end|>', '').strip()
            ref_action, ref_input = ReactORM.parse_output(reference)
            pred_action, pred_input = ReactORM.parse_output(prediction)
            action_ref.append(ref_action)
            action_input_ref.append(ref_input)
            if pred_action is None:
                action_pred.append('none')
            else:
                action_pred.append(pred_action)

            if pred_input is None:
                action_input_pred.append('{}')
            else:
                action_input_pred.append(pred_input)

            reward = ReactORM.evaluate_action_reward(action_pred, action_ref, action_input_pred, action_input_ref)
            rewards.append(float(reward))
        return rewards

    @staticmethod
    def evaluate_rougel(cand_list: list, ref_list: list):
        if len(ref_list) == 0:
            return None
        try:
            from rouge import Rouge
            rouge = Rouge()
            rouge_score = rouge.get_scores(hyps=cand_list, refs=ref_list, avg=True)
            rougel = rouge_score['rouge-l']['f']
            return rougel
        except Exception:
            return None


class MathORM(ORM):

    def __init__(self):
        from transformers.utils import strtobool
        self.use_opencompass = strtobool(os.environ.get('USE_OPENCOMPASS_EVALUATOR', 'False'))
        if self.use_opencompass:
            from opencompass.datasets.math import MATHEvaluator
            self.evaluator = MATHEvaluator()

    @staticmethod
    def check_terminate(answers: Union[str, List[str]]) -> List[bool]:
        if isinstance(answers, str):
            answers = [answers]
        results = []
        for answer in answers:
            results.append('\\boxed' in answer)
        return results

    @staticmethod
    def extract_boxed_result(text):
        pattern = r'\\boxed{([^}]*)}'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        else:
            return text

    @staticmethod
    def clean_latex(latex_str):
        latex_str = re.sub(r'\\\(|\\\)|\\\[|\\]', '', latex_str)
        latex_str = latex_str.replace('}}', '}').replace('{', '').replace('}', '')
        return latex_str.strip()

    @staticmethod
    def parse_expression(latex_str):
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        try:
            expr = parse_latex(latex_str)
            return simplify(expr)
        except Exception:
            return None

    @staticmethod
    def compare_consecutive(first, second):
        cleaned_list = [MathORM.clean_latex(latex) for latex in [first, second]]
        parsed_exprs = [MathORM.parse_expression(latex) for latex in cleaned_list]
        if hasattr(parsed_exprs[0], 'equals') and hasattr(parsed_exprs[1], 'equals'):
            value = parsed_exprs[0].equals(parsed_exprs[1])
        else:
            value = parsed_exprs[0] == parsed_exprs[1]
        if value is None:
            value = False
        return value

    def __call__(self, infer_requests: List[Union[InferRequest, Dict]], ground_truths: List[str],
                 **kwargs) -> List[float]:
        rewards = []
        predictions = [request.messages[-1]['content'] for request in infer_requests]
        for prediction, ground_truth in zip(predictions, ground_truths):
            if '# Answer' in prediction:
                prediction = prediction.split('# Answer')[1]
            if '# Answer' in ground_truth:
                ground_truth = ground_truth.split('# Answer')[1]
            prediction = prediction.strip()
            ground_truth = ground_truth.strip()
            prediction = MathORM.extract_boxed_result(prediction)
            ground_truth = MathORM.extract_boxed_result(ground_truth)
            if self.use_opencompass:
                reward = self.evaluator.is_equiv(prediction, ground_truth)
            else:
                reward = MathORM.compare_consecutive(prediction, ground_truth)
            rewards.append(float(reward))
        return rewards


class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            "The math_verify package is required but not installed. Please install it using 'pip install math_verify'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            gold_parsed = parse(sol, extraction_mode='first_match')
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed=True,
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode='first_match',
                )
                # edge case
                try:
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception:
                    reward = 0.0
            else:
                # If the gold solution is not parseable, we reward 0 to skip this example
                reward = 0.0
            rewards.append(reward)
        return rewards


class Format(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        # print(completions)
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class ReActFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*Action:.*?Action Input:.*?$'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CosineReward(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self,
                 tokenizer=None,
                 cosine_min_len_value_wrong: float = -0.5,
                 cosine_max_len_value_wrong: float = 0.0,
                 cosine_min_len_value_correct: float = 1.0,
                 cosine_max_len_value_correct: float = 0.5,
                 cosine_max_len: int = 1000,
                 accuracy_orm=None):
        self.tokenizer = tokenizer
        self.min_len_value_wrong = cosine_min_len_value_wrong
        self.max_len_value_wrong = cosine_max_len_value_wrong
        self.min_len_value_correct = cosine_min_len_value_correct
        self.max_len_value_correct = cosine_max_len_value_correct
        self.max_len = cosine_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracy()

    @staticmethod
    def cosfn(t, T, min_value, max_value):
        import math
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        rewards = []
        for content, acc_reward in zip(completions, acc_rewards):
            is_correct = acc_reward >= 1.
            if is_correct:
                # Swap min/max for correct answers
                min_value = self.max_len_value_correct
                max_value = self.min_len_value_correct
            else:
                min_value = self.max_len_value_wrong
                max_value = self.min_len_value_wrong
            gen_len = len(self.tokenizer.encode(content))
            reward = self.cosfn(gen_len, self.max_len, min_value, max_value)
            rewards.append(reward)
        return rewards


class RepetitionPenalty(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self, repetition_n_grams: int = 3, repetition_max_penalty: float = -1.0):
        self.ngram_size = repetition_n_grams
        self.max_penalty = repetition_max_penalty

    @staticmethod
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        reward function the penalizes repetitions

        Args:
            completions: List of model completions
        """
        rewards = []
        for completion in completions:
            if completion == '':
                rewards.append(0.0)
                continue
            if len(completion.split()) < self.ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in self.zipngram(completion, self.ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * self.max_penalty
            rewards.append(reward)
        return rewards


class SoftOverlong(ORM):

    def __init__(self, tokenizer, soft_max_length, soft_cache_length):
        self.tokenizer = tokenizer
        assert soft_cache_length < soft_max_length
        self.soft_max_length = soft_max_length
        self.soft_cache_length = soft_cache_length

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            completion_length = len(self.tokenizer.encode(completion))
            expected_len = self.soft_max_length - self.soft_cache_length
            exceed_len = completion_length - expected_len
            rewards.append(min(-exceed_len / self.soft_cache_length, 0))
        return rewards


import re
import numpy as np
from Levenshtein import distance as edit_distance

findt = re.compile(r'"(.*?)"')
findnum = re.compile(r'\d+')
search = re.compile(r"<answer>(.*?)<\/answer>")

# ipt = '<|object-bbox|> (29, 31), (34, 35) </|object-bbox|>'
# rst = findnum.findall(ipt)
# print(rst)
# print(search.search(ipt).group(1).strip())
# exit(0)
class RealFakeORM(ORM):
    def __init__(self):
        self.findt = re.compile(r'"(.*?)"')
        self.findn = re.compile(r'[^\w\s]')
        self.search = re.compile(r"<answer>(.*?)<\/answer>")
        self.findnum = re.compile(r'\d+')

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Evaluates completions based on Mathematical correctness of the answer
        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
            nums (list[str]): Available numbers
        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for completion, gt in zip(completions, solution):
            try:
                # Check if the format is correct
                match = re.search(r"<answer>(.*?)<\/answer>", completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                rst = match.group(1).strip()
                if gt=='<answer> This image is entirely generated. </answer>':
                    if ('is entirely generated' in rst) or ('is totally generated' in rst):
                        rewards.append(1.0)
                    else:
                        rewards.append(0.0)
                elif gt=='<answer> This image is entirely generated. </answer>':
                    if ('is real' in rst) or ('is authentic' in rst):
                        rewards.append(1.0)
                    else:
                        rewards.append(0.0)
                else:
                    if ('tampered' in rst):
                        rewards.append(1.0)
                    else:
                        rewards.append(0.0)
            except:# except Exception as e:
                # If evaluation fails, reward is 0
                print('RealFale error', completion, gt)
                rewards.append(0.0)
        return rewards


class MethodORM(ORM):
    def __init__(self):
        self.findt = re.compile(r'"(.*?)"')
        self.findn = re.compile(r'[^\w\s]')
        self.search = re.compile(r"<answer>(.*?)<\/answer>")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Evaluates completions based on Mathematical correctness of the answer
        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
            nums (list[str]): Available numbers
        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for completion, gt in zip(completions, solution):
            try:
                # Check if the format is correct
                match = self.search.search(completion) # re.search(r"<answer>(.*?)<\/answer>", completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                rst = match.group(1).strip()
                if (('tampered' in gt) and ('tampered' in rst)):
                    if 'copy-paste' in gt:
                        if 'copy-paste' in rst:
                            rewards.append(1.0)
                        else:
                            rewards.append(0.0)
                    else:
                        if 'generation' in rst:
                            rewards.append(1.0)
                        else:
                            rewards.append(0.0)
                else:
                    rewards.append(0.0)
            except:# except Exception as e:
                # If evaluation fails, reward is 0
                print('Method error', completion, gt)
                rewards.append(0.0)
        return rewards


class OCRORM(ORM):
    def __init__(self):
        self.findt = re.compile(r'"(.*?)"')
        self.findn = re.compile(r'[^\w\s]')
        self.search = re.compile(r"<answer>(.*?)<\/answer>")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Evaluates completions based on Mathematical correctness of the answer
        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
            nums (list[str]): Available numbers
        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for completion, gt in zip(completions, solution):
            try:
                # Check if the format is correct
                match = self.search.search(completion) # re.search(r"<answer>(.*?)<\/answer>", completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                rst = match.group(1).strip()
                if (('tampered' in gt) and ('tampered' in rst)):
                    fgt = self.findt.findall(gt)
                    fpt = self.findt.findall(rst)
                    # print(fgt, fpt)
                    gtext = fgt[0]
                    ptext = fpt[0]
                    if max(len(ptext), len(gtext))!=0:
                        reward1 = (1-edit_distance(ptext, gtext)/max(len(ptext), len(gtext)))
                    else:
                        reward1 = (0.0)
                    gtext = fgt[1]
                    ptext = fpt[1]
                    if max(len(ptext), len(gtext))!=0:
                        reward2 = (1-edit_distance(ptext, gtext)/max(len(ptext), len(gtext)))
                    else:
                        reward2 = (0.0)
                    rewards.append(reward1 + reward2)
                else:
                    rewards.append(0.0)
            except:# except Exception as e:
                # If evaluation fails, reward is 0
                print('OCR error', completion, gt)
                rewards.append(0.0)
        return rewards



class RepORM(ORM):
    def __init__(self):
        self.findt = re.compile(r'"(.*?)"')
        self.findn = re.compile(r'[^\w\s]')
        self.search = re.compile(r"<answer>(.*?)<\/answer>")
        self.think =  re.compile(r"<think>(.*?)<\/think>")

    def solves(self, tokens):
        max_n = 3
        best_ngram = []
        best_count = 0
        best_n = 0
        L = len(tokens)
        for n in range(1, min(max_n, L) + 1):
            i = 0
            while i + n <= L:
                ngram = tuple(tokens[i:i+n])
                cnt = 1
                j = i + n
                while j + n <= L and tuple(tokens[j:j+n]) == ngram:
                    cnt += 1
                    j += n
                if cnt > best_count or (cnt == best_count and n > best_n):
                    best_ngram = list(ngram)
                    best_count = cnt
                    best_n = n
                i = j if cnt > 1 else i + 1
        return best_count

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Evaluates completions based on Mathematical correctness of the answer
        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
            nums (list[str]): Available numbers
        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for completion, gt in zip(completions, solution):
            try:
                print(completions)
                # Check if the format is correct
                match = self.think.search(completion) # re.search(r"<answer>(.*?)<\/answer>", completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                pred = match.group(1).strip()
                ptext = self.findt.findall(pred)
                for p in ptext:
                    pred = pred.replace(p, '')
                pred = self.findn.sub('', pred)
                max_nums = self.solves(pred.split(' '))
                # print(pred.split(' '), max_nums)
                if max_nums>=8:
                    rewards.append(-10.0)
                else:
                    rewards.append(0.0)
            except:# except Exception as e:
                # If evaluation fails, reward is 0
                print('Rep error', completion, gt)
                rewards.append(0.0)
        print('R', len(completions), len(solution), rewards)
        return rewards


class IoUORM(ORM):
    def __init__(self):
        self.findt = re.compile(r'"(.*?)"')
        self.findn = re.compile(r'[^\w\s]')
        self.search = re.compile(r"<answer>(.*?)<\/answer>")
        self.findnum = re.compile(r'\d+')

    def calculate_l1(self, box1, box2):
        return np.array([np.abs(box2[i]-box1[i]) for i in range(4)]).mean()

    def calculate_iou(self, box1, box2):
        # box1: [x_min, y_min, x_max, y_max]
        # box2: [x_min, y_min, x_max, y_max]
        
        # 计算交集区域的坐标
        x_inter_min = np.maximum(box1[0], box2[0])
        y_inter_min = np.maximum(box1[1], box2[1])
        x_inter_max = np.minimum(box1[2], box2[2])
        y_inter_max = np.minimum(box1[3], box2[3])
        
        # 计算交集面积
        intersection_area = np.maximum(x_inter_max - x_inter_min, 0) * np.maximum(y_inter_max - y_inter_min, 0)
        
        # 计算两个框的面积
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # 计算并集面积
        union_area = area1 + area2 - intersection_area
        
        # 计算IoU
        iou = intersection_area / union_area
        return iou
 
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Evaluates completions based on Mathematical correctness of the answer
        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
            nums (list[str]): Available numbers
        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for completion, gt in zip(completions, solution):
            try:
                # Check if the format is correct
                match = self.search.search(completion) # re.search(r"<answer>(.*?)<\/answer>", completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                rst = match.group(1).strip()
                if (('tampered' in gt) and ('tampered' in rst)):
                    # print(rst, gt)
                    gt_num = [int(x) for x in self.findnum.findall(gt)]
                    pred_num = [int(x) for x in self.findnum.findall(rst)]
                    if len(pred_num)!=4:
                        rewards.append(0.0)
                    else:
                        iou = self.calculate_iou(pred_num, gt_num)
                        if iou<0.5:
                            iou_reward = 0.0
                        elif iou>0.8:
                            iou_reward = 1.0
                        else:
                            iou_reward = iou
                        l1 = self.calculate_l1(pred_num, gt_num)
                        if l1<=10:
                            l1_reward = 1.0
                        else:
                            l1_reward = 0.0
                        rewards.append(iou_reward + l1_reward)
                else:
                    rewards.append(0.0)
            except:# except Exception as e:
                # If evaluation fails, reward is 0
                print('IoU error', completion, gt)
                rewards.append(0.0)
        return rewards

orms = {
    'toolbench': ReactORM,
    'math': MathORM,
    'accuracy': MathAccuracy,
    'format': Format,
    'react_format': ReActFormat,
    'cosine': CosineReward,
    'repetition': RepetitionPenalty,
    'soft_overlong': SoftOverlong,
}

orms['realfake'] = RealFakeORM
orms['method'] = MethodORM
orms['ocr'] = OCRORM
orms['iou'] = IoUORM
orms['rep'] = RepORM
