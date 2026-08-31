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
import pickle
from tqdm import tqdm
import json, logging

T = TypeVar("T")
PROMPTS_ROOT = pathlib.Path('ConRAG/prompts').resolve()
logger = logging.getLogger()


def get_random_tokens(input_str, tokenizer):
    if type(input_str) == str:
        input_ids = tokenizer.encode(input_str, add_special_tokens=False)
        n = len(input_ids) - 1
    elif type(input_str) == int:
        n = input_str
    else:
        raise ValueError(input_str)
    vocab_size = tokenizer.vocab_size
    random_ids = [random.randint(0, vocab_size - 1) for _ in range(n)]
    random_str = tokenizer.decode(random_ids, skip_special_tokens=True)
    return random_str

def get_qa_instruction(
    question: str, paragraphs: List, tokenizer, retrieval_aware: bool, RETRIEVAL_TOKEN, use_cot, prompt_type, add_noise, noise_type, gt_position, noise_num, standard_prompt, retrieval_aware_type, num_k):
    if not question:
        raise ValueError(f"Provided `question` must be truthy, got: {question}")
    if not paragraphs:
        raise ValueError(f"Provided `paragraphs` must be truthy, got: {paragraphs}")

    if retrieval_aware and not standard_prompt:
        prompt_filename = 'qa_similarity.prompt'
    elif use_cot:
        prompt_filename = 'qa_cot.prompt'
    elif prompt_type != '':
        prompt_filename = f'{prompt_type}.prompt'
    else:
        prompt_filename = "qa.prompt"

    with open(PROMPTS_ROOT / prompt_filename) as f:
        prompt_template = f.read().rstrip("\n")
    
    formatted_documents = []
    if not add_noise:
        for document_index, document in enumerate(paragraphs[:num_k]):
            if retrieval_aware and not standard_prompt:
                document_prompt = f"[{document_index+1}]similarity: {RETRIEVAL_TOKEN}(Title: {document['title']}) {document['paragraph_text']}"
            elif 'mask' in prompt_type:
                document_prompt = f"{RETRIEVAL_TOKEN}(Title: {document['title']}) {document['paragraph_text']}"
            elif retrieval_aware and standard_prompt:
                document_prompt = f"[{document_index+1}]{RETRIEVAL_TOKEN}(Title: {document['title']}) {document['paragraph_text']}"
            else:
                document_prompt = f"[{document_index+1}](Title: {document['title']}) {document['paragraph_text']}"
            formatted_documents.append(document_prompt)
    else:
        raise ValueError()
        # documents_text = []
        # for document_index, document in enumerate(paragraphs):
        #     if noise_type == 'random_tokens' and not document['is_supporting']:
        #         document_prompt = f"[{document_index+1}](Title: {get_random_tokens(document['title'], tokenizer)}) {get_random_tokens(document['paragraph_text'], tokenizer)}"
        #     elif noise_type == 'add_random_tokens':
        #         document_prompt = f"[{document_index+1}](Title: {document['title']}) {document['paragraph_text']} {get_random_tokens(int(noise_num), tokenizer)}"
        #     elif noise_type == 'add_random_tokens_distractors' and not document['is_supporting']:
        #         document_prompt = f"[{document_index+1}](Title: {document['title']}) {document['paragraph_text']} {get_random_tokens(int(noise_num), tokenizer)}"
        #     elif noise_type == 'replace_random_tokens' and not document['is_supporting']:
        #         document_prompt = f"[{document_index+1}]{get_random_tokens(int(noise_num), tokenizer)}"
        #     else:
        #         document_prompt = f"[{document_index+1}](Title: {document['title']}) {document['paragraph_text']}"
        #     documents_text.append((document_prompt, document['is_supporting']))
        # gold_documents = [doc for doc, isgold in documents_text if isgold]
        # noise_documents = [doc for doc, isgold in documents_text if not isgold]
        # if gt_position == 'default':
        #     formatted_documents.extend([doc for doc, isgold in documents_text])
        # elif gt_position == 'far':
        #     formatted_documents.extend(gold_documents + noise_documents)
        # elif gt_position == 'mid':
        #     mid_index = len(noise_documents) // 2
        #     formatted_documents.extend(noise_documents[:mid_index] + gold_documents + noise_documents[mid_index:])
        # elif gt_position == 'near':
        #     formatted_documents.extend(noise_documents + gold_documents)
        # elif gt_position == 'none':
        #     formatted_documents.extend(gold_documents)
    if retrieval_aware_type != '' and retrieval_aware:
        raise ValueError
    if retrieval_aware_type == 'far':
        return prompt_template.format(question=question, search_results= RETRIEVAL_TOKEN*num_k + "\n".join(formatted_documents))
    if retrieval_aware_type == 'near':
        return prompt_template.format(question=question, search_results= "\n".join(formatted_documents) + RETRIEVAL_TOKEN*num_k)
    if retrieval_aware_type == 'prompt_tuning':
        return RETRIEVAL_TOKEN*num_k + prompt_template.format(question=question, search_results= "\n".join(formatted_documents))
    if retrieval_aware_type == 'consistent':
        return RETRIEVAL_TOKEN*num_k + prompt_template.format(question=question, search_results= "".join(formatted_documents))
    if 'question2' in prompt_type:
        return prompt_template.format(question2=question, question=question, search_results="\n".join(formatted_documents))

    return prompt_template.format(question=question, search_results="\n".join(formatted_documents))


def get_instruction_dataset(dataset, max_prompt_length, tokenizer, retrieval_aware, use_cot, prompt_type, RETRIEVAL_TOKEN, add_noise, noise_type, gt_position, noise_num, standard_prompt, retrieval_aware_type, num_k, sample_answer=True):
    instruction_dataset = []
    for input_example in tqdm(dataset):
        input_example = deepcopy(input_example)
        question = input_example["question"]
        paragraphs = input_example["paragraphs"]
        if not paragraphs:
            raise ValueError(f"Did not find any documents for example: {input_example}")
        prompt = get_qa_instruction(
                question,
                paragraphs,
                tokenizer=tokenizer,
                retrieval_aware=retrieval_aware,
                use_cot=use_cot,
                prompt_type=prompt_type,
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

        answers = [input_example['answer']] if sample_answer else [input_example['answer']] + input_example['answer_aliases']
        for ans in answers:
            prompt_length = len(tokenizer(prompt + ans)["input_ids"])
            if max_prompt_length < prompt_length:
                print(
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
        documents = data['paragraphs'][:num_k]
        data['embeds'] = [d['emb'] for d in documents] if use_emb else [[float(d['rerank_score']), float(d['rerank_nb_score']), float(d['rerank_precedent_score'])] for d in documents]
        data['label'] = [1 if d['is_supporting'] else 0 for d in documents]
    return dataset

def load_musique_data(input_path):
    train_data_path = input_path['train_data_path']
    test_data_path = input_path['test_data_path']
    with open(train_data_path, 'rb') as f:
        train_data = pickle.load(f)[:]
        f.close()
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)[:]
        f.close()
    print(f'prepare dataset, train size: {len(train_data)}, test size: {len(test_data)}')
    return train_data, test_data


def load_compress_data(input_path):
    test_data_path = input_path['test_data_path']
    if test_data_path.endswith('.json'):
        examples = json.load(open(test_data_path))
        dataset = []
        for example in examples.values():
            example['instruction'] = example['compressed_prompt']
            dataset.append(example)
    elif test_data_path.endswith('.jsonl'):
        dataset =[]
        prompt_filename = "qa.prompt"
        with open(PROMPTS_ROOT / prompt_filename) as f:
            prompt_template = f.read().rstrip("\n")

        with open(test_data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                data = json.loads(line.strip())
                data['instruction'] = prompt_template.format(question=data['question'], search_results=data['compressed_prompt'])
                dataset.append(data)
            f.close()
    elif test_data_path.endswith('.pkl'):
        examples = pickle.load(open(test_data_path, 'rb'))
        dataset = []
        for example in examples:
            example['instruction'] = example['compressed_prompt']['compressed_prompt']
            dataset.append(example)
    return dataset

def load_musique_dataset(input_path, max_prompt_length, tokenizer, retrieval_aware, use_cot=False, prompt_type='', RETRIEVAL_TOKEN='<R>', add_noise=False, noise_type='', gt_position='', noise_num=100, use_emb=True, standard_prompt=False, retrieval_aware_type='', num_k=20, ignore_train=False):
    if 'compress' in input_path['test_data_path']:
        return None, load_compress_data(input_path)
    train_data, test_data = load_musique_data(input_path)
    instruction_dataset_train = get_instruction_dataset(train_data[:], max_prompt_length, tokenizer, retrieval_aware, use_cot, prompt_type, RETRIEVAL_TOKEN, add_noise, noise_type, gt_position, noise_num, standard_prompt, retrieval_aware_type, num_k) if not ignore_train else None
    instruction_dataset_test = get_instruction_dataset(test_data[:], max_prompt_length, tokenizer, retrieval_aware, use_cot, prompt_type, RETRIEVAL_TOKEN, add_noise, noise_type, gt_position, noise_num, standard_prompt, retrieval_aware_type, num_k)

    instruction_dataset_train = get_embeds(instruction_dataset_train, use_emb, num_k) if not ignore_train else None
    instruction_dataset_test = get_embeds(instruction_dataset_test, use_emb, num_k)
    return instruction_dataset_train, instruction_dataset_test

def get_musique_ans(dataset):
    if 'answers' in dataset[0].keys():
        return [data['answers'] for data in dataset]
    else:
        return [[data['answer']]+data['answer_aliases'] for data in dataset]


