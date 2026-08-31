import torch
import numpy as np
import random, os
def seed_it(seed):
    os.environ["PYTHONSEED"] = str(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


from transformers import AutoTokenizer
from typing import List, Optional, Tuple, Type, TypeVar
from copy import deepcopy
from pydantic.dataclasses import dataclass
import pathlib

T = TypeVar("T")

PROMPTS_ROOT = pathlib.Path('ConRAG/prompts').resolve()
    
import pickle
from tqdm import tqdm
import json, logging


logger = logging.getLogger()
T = TypeVar("T")

def get_qa_instruction(
    question: str, documents: List, tokenizer, retrieval_aware: bool, RETRIEVAL_TOKEN, use_cot, add_noise, noise_type, gt_position, noise_num, standard_prompt, retrieval_aware_type, num_k):
    if not question:
        raise ValueError(f"Provided `question` must be truthy, got: {question}")
    if not documents:
        raise ValueError(f"Provided `documents` must be truthy, got: {documents}")

    if retrieval_aware and not standard_prompt:
        prompt_filename = 'qa_similarity_zh.prompt'
    elif use_cot:
        prompt_filename = 'qa_cot_zh.prompt'
    else:
        prompt_filename = "qa_zh.prompt"

    with open(PROMPTS_ROOT / prompt_filename) as f:
        prompt_template = f.read().rstrip("\n")
    
    formatted_documents = []
    if not add_noise:
        for document_index, document in enumerate(documents[:num_k]):
            if retrieval_aware and not standard_prompt:
                document_prompt = f"[{document_index+1}]similarity: {RETRIEVAL_TOKEN}{document['text']}"
            elif retrieval_aware and standard_prompt:
                document_prompt = f"[{document_index+1}]{RETRIEVAL_TOKEN}{document['text']}"
            else:
                document_prompt = f"[{document_index+1}]{document['text']}"
            formatted_documents.append(document_prompt)
    else:
        raise ValueError()
    if retrieval_aware_type == 'prompt_tuning':
        return RETRIEVAL_TOKEN*num_k + prompt_template.format(question=question, search_results= "\n".join(formatted_documents))
    return prompt_template.format(question=question, search_results="\n".join(formatted_documents))


def get_instruction_dataset(dataset, idx, max_prompt_length, tokenizer, retrieval_aware, use_cot, RETRIEVAL_TOKEN, add_noise, noise_type, gt_position, noise_num, standard_prompt, retrieval_aware_type, num_k, sample_answer=True):
    instruction_dataset = []
    for i in tqdm(idx):
        input_example = deepcopy(dataset[i])
        question = input_example["question"]
        documents = []
        for ctx in deepcopy(input_example["ctxs"]):
            documents.append(ctx)
        if len(documents) != 10:
            continue
        if not documents:
            raise ValueError(f"Did not find any documents for example: {input_example}")
        prompt = get_qa_instruction(
                question,
                documents,
                tokenizer=tokenizer,
                retrieval_aware=retrieval_aware,
                use_cot=use_cot,
                RETRIEVAL_TOKEN=RETRIEVAL_TOKEN,
                add_noise=add_noise,
                noise_type=noise_type,
                gt_position=gt_position,
                noise_num=noise_num,
                standard_prompt=standard_prompt,
                retrieval_aware_type=retrieval_aware_type,
                num_k=num_k,
            )
        
        input_example['instruction'] = prompt

        answers = [input_example['answer']]
        for ans in answers:
            prompt_length = len(tokenizer(prompt + ans)["input_ids"])
            if max_prompt_length < prompt_length:
                logger.info(
                            f"Skipping prompt ... with length {prompt_length}, which "
                            f"is greater than maximum prompt length {max_prompt_length}"
                )
                continue
            data = deepcopy(input_example)
            data['output'] = ans
            instruction_dataset.append(data)
    return instruction_dataset

def get_embeds(dataset, use_emb, num_k):
    for data in dataset:
        documents = data['ctxs'][:num_k]
        data['embeds'] = [d['emb'] for d in documents] if use_emb else [[float(d['rerank_score']), float(d['rerank_nb_score']), float(d['rerank_precedent_score'])] for d in documents]
        data['label'] = [1 if d['label'] else 0 for d in documents]
    return dataset

def load_lkqa_data(input_path, dataset_seed=42):
    with open(input_path, 'rb') as fin:
        examples = pickle.load(fin)
        fin.close()
    seed_it(dataset_seed)
    all_index = list(range(len(examples)))
    train_index = np.sort(random.sample(all_index, int(len(all_index)*0.8)))
    test_index = np.sort(list(set(all_index).difference(set(train_index))))
    print(f'prepare dataset, train size: {len(train_index)}, test size: {len(test_index)}')
    return examples, train_index, test_index


def load_lkqa_dataset(input_path, max_prompt_length, tokenizer, retrieval_aware=True, use_cot=False, RETRIEVAL_TOKEN='<R>', add_noise=False, noise_type='', gt_position='', noise_num=100, use_emb=True, standard_prompt=False, retrieval_aware_type='', num_k=10, dataset_seed=42, ignore_train=False):
    examples, train_index, test_index = load_lkqa_data(input_path, dataset_seed)
    instruction_dataset_train = get_instruction_dataset(examples, train_index, max_prompt_length, tokenizer, retrieval_aware, use_cot, RETRIEVAL_TOKEN, add_noise, noise_type, gt_position, noise_num, standard_prompt, retrieval_aware_type, num_k)
    instruction_dataset_test = get_instruction_dataset(examples, test_index, max_prompt_length, tokenizer, retrieval_aware, use_cot, RETRIEVAL_TOKEN, add_noise, noise_type, gt_position, noise_num, standard_prompt, retrieval_aware_type, num_k)

    instruction_dataset_train = get_embeds(instruction_dataset_train, use_emb, num_k)
    instruction_dataset_test = get_embeds(instruction_dataset_test, use_emb, num_k)
    return instruction_dataset_train, instruction_dataset_test

def get_lkqa_ans(dataset):
    if 'answers' in dataset[0].keys():
        return [data['answers'] for data in dataset]
    else:
        return [[data['answer']] for data in dataset]