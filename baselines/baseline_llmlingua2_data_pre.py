import torch
import numpy as np
import random, os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# torch.cuda.set_device(0)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)
seed_it(42)


# nohup python llmlingua_data_pre.py >  logs/llmlingua/output_data_pre_hotpotqa_llama2.log 2>&1 &
# nohup python llmlingua_data_pre.py >  logs/llmlingua/output_data_pre_2wiki_llama2.log 2>&1 &
# nohup python llmlingua_data_pre.py >  logs/llmlingua/output_data_pre_musique_llama2.log 2>&1 &

# nohup python llmlingua_data_pre.py >  logs/llmlingua/output_data_pre_musique_mistral.log 2>&1 &
# nohup python llmlingua_data_pre.py >  logs/llmlingua/output_data_pre_nq20_mistral.log 2>&1 &
# nohup python llmlingua_data_pre.py >  logs/llmlingua/output_data_pre_nq20_llama2.log 2>&1 &

# nohup python llmlingua_data_pre.py >  logs/llmlingua/output_data_pre_mrag_llama2.log 2>&1 &
# nohup python llmlingua_data_pre.py >  logs/llmlingua/output_data_pre_mrag_mistral.log 2>&1 &



# dataset_name = '2wiki'
# dataset_name = 'musique'
# dataset_name = 'nq20'
dataset_name = 'mrag'
# llama2 = '_llama2'
llama2 = ''

# at least two keys: *idx* and *prompt*. 
from transformers import AutoTokenizer
# model_name = 
# model_name = 'models/longchat-7b-v1.5-32k'
model_name = 'models/Mistral-7B-Instruct-v0.3' if llama2 == '' else 'models/Llama-2-7b-chat-hf'

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

import sys
import os


if dataset_name == 'nq20':
    from ConRAG.dataset.load_nq import *
    instruction_dataset_train, instruction_dataset_test = load_nq_dataset(
        'noise/ConRAG/dataset/nq/nq-open-20_total_documents_gold_bert.pkl', 
        4096,
        tokenizer,
        retrieval_aware=False,
        num_k=20,
        use_cot=False, 
        add_noise=False, 
        noise_type='', 
        gt_position='none',
        use_emb=False,
    )
    seed_it(42)
    all_index = list(range(len(instruction_dataset_train) + len(instruction_dataset_test)))
    # all_index = list(range(20))
    # train_index = np.sort(random.sample(all_index, int(len(all_index)*0.8)))
    train_index = np.sort(random.sample(all_index, int(len(all_index)*0.8)))
    test_index = np.sort(list(set(all_index).difference(set(train_index))))
    data = [{
        'idx': str(test_index[i]),
        'prompt': instruction_dataset_test[i]['instruction'],
        'question': instruction_dataset_test[i]['question'],
        'answers': instruction_dataset_test[i]['answers'],
    } for i in range(len(test_index))]
    print(f"num data: {len(data)}")
    data = data + [{
        'idx': str(train_index[i]),
        'prompt': instruction_dataset_train[i]['instruction'],
        'question': instruction_dataset_train[i]['question'],
        'answers': instruction_dataset_train[i]['answers'],
    } for i in range(len(train_index))]
    print(f"num data: {len(data)}")

if dataset_name == 'mrag':
    from ConRAG.dataset.load_mrag import *
    instruction_dataset_train, instruction_dataset_test = load_mrag_dataset(
        'noise/ConRAG/dataset/multihop_rag/multihop_rag_emb.pkl', 
        32000,
        tokenizer,
        retrieval_aware=False,
        num_k=10,
        use_cot=False, 
        add_noise=False, 
        noise_type='', 
        gt_position='none',
        use_emb=False,
    )
    print('len', len(instruction_dataset_train), len(instruction_dataset_test))
    seed_it(42)
    all_index = list(range(len(instruction_dataset_train) + len(instruction_dataset_test)))
    # all_index = list(range(20))
    # train_index = np.sort(random.sample(all_index, int(len(all_index)*0.8)))
    train_index = np.sort(random.sample(all_index, int(len(all_index)*0.8)))
    test_index = np.sort(list(set(all_index).difference(set(train_index))))
    data = [{
        'idx': str(i),
        'prompt': instruction_dataset_test[i]['instruction'],
        'question': instruction_dataset_test[i]['query'],
        'answers': [instruction_dataset_test[i]['answer']],
    } for i in range(len(instruction_dataset_test))]
    print(f"num data: {len(data)}")
    data = data + [{
        'idx': str(i+len(instruction_dataset_test)),
        'prompt': instruction_dataset_train[i]['instruction'],
        'question': instruction_dataset_train[i]['query'],
        'answers': [instruction_dataset_train[i]['answer']],
    } for i in range(len(instruction_dataset_train))]
    print(f"num data: {len(data)}")

if dataset_name == 'hotpotqa':
    from ConRAG.dataset.load_hotpotqa import *
    input_path = {
        'train_data_path': 'dataset/hotpotqa/hotpot_train_v1.1_bert.pkl',
        'test_data_path': 'dataset/hotpotqa/hotpot_dev_distractor_v1_bert.pkl',
        }
    max_prompt_length = 32000
    instruction_dataset_train, instruction_dataset_test = load_hotpotqa_dataset(
        input_path=input_path,
        max_prompt_length=max_prompt_length,
        tokenizer=tokenizer,
        retrieval_aware=False,
        use_cot=False,
        add_noise=False,
        ignore_train=False,
        )
    data = [{
        'idx': str(i),
        'prompt': instruction_dataset_test[i]['instruction'],
        'question': instruction_dataset_test[i]['question'],
        'answers': [instruction_dataset_test[i]['answer']],
    } for i in range(len(instruction_dataset_test))]
    print(f"num data: {len(data)}")
    data = data + [{
        'idx': str(i + len(instruction_dataset_test)),
        'prompt': instruction_dataset_train[i]['instruction'],
        'question': instruction_dataset_train[i]['question'],
        'answers': [instruction_dataset_train[i]['answer']],
    } for i in range(len(instruction_dataset_train))]
    print(f"num data: {len(data)}")

if dataset_name == '2wiki':
    from ConRAG.dataset.load_hotpotqa import *
    input_path = {
        'train_data_path': 'dataset/2wikimultihop/2wikimultihop_train_bert.pkl',
        'test_data_path': 'dataset/2wikimultihop/2wikimultihop_dev_bert.pkl',
        }
    max_prompt_length = 32000
    instruction_dataset_train, instruction_dataset_test = load_hotpotqa_dataset(
        input_path=input_path,
        max_prompt_length=max_prompt_length,
        tokenizer=tokenizer,
        retrieval_aware=False,
        use_cot=False,
        add_noise=False,
        ignore_train=False,
        )
    data = [{
        'idx': str(i),
        'prompt': instruction_dataset_test[i]['instruction'],
        'question': instruction_dataset_test[i]['question'],
        'answers': [instruction_dataset_test[i]['answer']],
    } for i in range(len(instruction_dataset_test))]
    print(f"num data: {len(data)}")
    data = data + [{
        'idx': str(i + len(instruction_dataset_test)),
        'prompt': instruction_dataset_train[i]['instruction'],
        'question': instruction_dataset_train[i]['question'],
        'answers': [instruction_dataset_train[i]['answer']],
    } for i in range(len(instruction_dataset_train))]
    print(f"num data: {len(data)}")

if dataset_name == 'musique':
    from ConRAG.dataset.load_musique import *
    train_data_path = 'noise/ConRAG/dataset/musique/musique_ans_v1.0_train_bert.pkl'
    test_data_path = 'noise/ConRAG/dataset/musique/musique_ans_v1.0_dev_bert.pkl'
    instruction_dataset_train, instruction_dataset_test = load_musique_dataset(
        {'train_data_path': train_data_path, 'test_data_path': test_data_path}, 
        retrieval_aware=False,
        use_cot=False,
        use_emb=False,
        standard_prompt=False,
        num_k=20,
        max_prompt_length=4096, tokenizer=tokenizer, ignore_train=False)
    data = [{
        'idx': str(i),
        'prompt': instruction_dataset_test[i]['instruction'],
        'question': instruction_dataset_test[i]['question'],
        'answers': [instruction_dataset_test[i]['answer']]+instruction_dataset_test[i]['answer_aliases'],
    } for i in range(len(instruction_dataset_test))]
    print(f"num data: {len(data)}")
    data = data + [{
        'idx': str(i + len(instruction_dataset_test)),
        'prompt': instruction_dataset_train[i]['instruction'],
        'question': instruction_dataset_train[i]['question'],
        'answers': [instruction_dataset_train[i]['answer']]+instruction_dataset_train[i]['answer_aliases'],
    } for i in range(len(instruction_dataset_train))]
    print(f"num data: {len(data)}")

import sys
import os
import argparse
import copy
import json
import os
import time

import tiktoken
from tqdm import tqdm
sys.path.append('baselines/LLMLingua/experiments/llmlingua2/data_collection/')
from GPT4_compressor import PromptCompressor
prompts = json.load(open('baselines/LLMLingua/experiments/llmlingua2/data_collection/compression_instructions.json'))
prompt_id = 4
n_max_new_token = 4000
compression_rate = 0.5
n_target_token = -1
chunk_size = 512
compressor_name = "gpt4"
save_key = "compressed_prompt"
model_name = 'gpt-4o-mini'
load_key = "prompt"
system_prompt = prompts[str(prompt_id)]["system_prompt"]
user_prompt = prompts[str(prompt_id)]["user_prompt"]
compressor = PromptCompressor(
    model_name=model_name, system_prompt=system_prompt, user_prompt=user_prompt
)
results = {}
results_list = []
total_time = 0
# save_path = 'output/longllmlingua/hotpotqa_compression_prompts.json'
# if os.path.exists(save_path):
#     results = json.load(open(save_path))

tokenizer = tiktoken.encoding_for_model(model_name)
def chunk_origin(origin_text):
    origin_list = []
    origin_token_ids = tokenizer.encode(origin_text)
    end_token_ids = set(tokenizer.encode(".") + tokenizer.encode("\n"))
    n = len(origin_token_ids)
    st = 0
    while st < n:
        if st + chunk_size > n - 1:
            chunk = tokenizer.decode(origin_token_ids[st:n])
            origin_list.append(chunk)
            break
        else:
            ed = st + chunk_size
            for j in range(0, ed - st):
                if origin_token_ids[ed - j] in end_token_ids:
                    ed = ed - j
                    break
            chunk = tokenizer.decode(origin_token_ids[st : ed + 1])
            origin_list.append(chunk)
            st = ed + 1
    return origin_list
for sample in tqdm(data[:]):
    idx = str(sample["idx"])
    origin = copy.deepcopy(sample[load_key])
    if origin is None:
        raise ValueError(origin)
    if idx in results or str(idx) in results:
        # print(f"{idx}-th sample is processed")
        continue

    t = time.time()
    if compressor_name == "llmlingua" or compressor_name == "longllmlingua":
        raise ValueError
    else:
        # multi document
        if isinstance(origin, list):
            if chunk_size > 0:
                chunk_list = []
                for j, document in enumerate(origin):
                    ori_list = chunk_origin(document)
                    chunk_list.extend(ori_list)
                origin = chunk_list
        # single document
        else:
            origin = [origin]
            if chunk_size > 0:
                origin = chunk_origin(origin[0])
        # print(f"num chunk: {len(origin)}")
        comp_list = []
        for j, chunk in enumerate(origin):
            comp = compressor.query_template(chunk, n_max_new_token)
            comp_list.append(comp)
        assert len(origin) == len(comp_list)

    total_time += time.time() - t
    new_sample = copy.deepcopy(sample)
    new_sample["instruction_prompt"] = comp
    if (
        not (compressor_name == "llmlingua" or compressor_name == "longllmlingua")
        and len(comp_list) > 0
    ):
        assert len(origin) == len(comp_list)
        new_sample["instruction_list"] = comp_list[:]
        new_sample["prompt_list"] = origin[:]

    results[idx] = new_sample

# print(save_path, total_time)
# json.dump(
#     results, open(save_path, "w", encoding="utf8"), indent=4, ensure_ascii=False
# )
from transformers import AutoTokenizer
# model_name = 'models/Mistral-7B-Instruct-v0.3'
model_name = 'models/Llama-2-7b-chat-hf'

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
all_prompts = []
for item in results.values():
    all_prompts.extend(item["instruction_list"])
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

model = LLM(
        model=model_name,
        tensor_parallel_size=1,
        trust_remote_code=True,
        # download_dir=hf_cache_path,
        load_format="auto",
        # max_num_batched_tokens=32768,
    )
def format_prompt(messages):
    prompt = tokenizer.decode(tokenizer.apply_chat_template(messages))
    return prompt
# format_prompt(results["0"]['prompt_list'][0])
from tqdm import tqdm
def process_prompts(prompts):
    return [format_prompt(prompt) for prompt in tqdm(prompts)]
processed_prompts = process_prompts(all_prompts)
len(processed_prompts)
sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100, skip_special_tokens=False)
raw_responses = model.generate(processed_prompts[:], sampling_params)
responses = [output.outputs[0].text.strip('<|start_header_id|>assistant<|end_header_id|>').strip() for output in raw_responses]
index = 0
for item in results.values():
    num_prompts = len(item["instruction_list"])
    item["compressed_prompt_list"] = responses[index:index + num_prompts]
    index += num_prompts
results['0']['compressed_prompt_list']


save_path = f'output/longllmlingua/{dataset_name}_compression_prompts{llama2}.json'
print(save_path)
json.dump(
    results, open(save_path, "w", encoding="utf8"), indent=4, ensure_ascii=False
)

import argparse
import json
import logging
import os
from collections import defaultdict
from datasets import load_dataset
import spacy
import torch
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")
def split_string(input_string, ignore_tokens=set([","])):
    doc = nlp(input_string)
    word_list = []
    for word in doc:
        if word.lemma_ not in ignore_tokens:
            word_list.append(word.lemma_)
    return word_list

def is_equal(token1, token2):
    return token1.lower() == token2.lower()

origins, comps = [], []
for i, sample in results.items():
    if len(sample["prompt_list"]) != len(sample["compressed_prompt_list"]):
        print(f"{i}-th length not equal")
        continue
    origins.extend(sample["prompt_list"])
    comps.extend(sample["compressed_prompt_list"])
    
res = {}
res_pt = defaultdict(list)

num_sample = 0
compression_rate_avg = 0
find_rate_avg = 0
variation_rate_avg = 0
matching_rate_avg = 0
hitting_rate_avg = 0
alignment_gap_avg = 0

window_size = 400
for chunk_idx, (origin, comp) in tqdm(enumerate(zip(origins, comps))):
    num_sample += 1
    origin_tokens = split_string(origin)
    comp_tokens = split_string(comp)
    origin_tokens_set = set(origin_tokens)
    for token in origin_tokens:
        origin_tokens_set.add(token.lower())

    num_find = 0
    prev_idx = 0
    back_cnt = 0
    num_origin_tokens = len(origin_tokens)
    labels = [False] * num_origin_tokens
    for token in comp_tokens:
        flag = False
        if token in origin_tokens_set or token.lower() in origin_tokens_set:
            num_find += 1
        for i in range(window_size):
            # look forward
            token_idx = min(prev_idx + i, num_origin_tokens - 1)
            if is_equal(origin_tokens[token_idx], token) and not labels[token_idx]:
                labels[token_idx] = True
                # window do not go too fast
                if token_idx - prev_idx > window_size // 2:
                    prev_idx += window_size // 2
                else:
                    prev_idx = token_idx
                # if args.verbose:
                #     print(
                #         token,
                #         token_idx,
                #         prev_idx,
                #         origin_tokens[token_idx - 1 : token_idx + 2],
                #     )
                flag = True
                break
            # look backward
            token_idx = max(prev_idx - i, 0)
            if is_equal(origin_tokens[token_idx], token) and not labels[token_idx]:
                labels[token_idx] = True
                prev_idx = token_idx
                # if args.verbose:
                #     print(
                #         token,
                #         token_idx,
                #         prev_idx,
                #         origin_tokens[token_idx - 1 : token_idx + 2],
                #     )
                flag = True
                break

    retrieval_tokens = []
    for idx, token in enumerate(origin_tokens):
        if labels[idx]:
            retrieval_tokens.append(token)
    retrieval = " ".join(retrieval_tokens)

    comp_rate = len(comp_tokens) / len(origin_tokens)
    if len(comp_tokens) > 0:
        find_rate = num_find / len(comp_tokens)
    else:
        find_rate = 0.0
    variation_rate = 1 - find_rate
    hitting_rate = num_find / len(origin_tokens)
    matching_rate = sum(labels) / len(labels)
    alignment_gap = hitting_rate - matching_rate

    compression_rate_avg += comp_rate
    find_rate_avg += find_rate
    variation_rate_avg += variation_rate
    hitting_rate_avg += hitting_rate
    matching_rate_avg += matching_rate
    alignment_gap_avg += alignment_gap

    # if alignment_gap > 0.1:
    #     print(origin)
    #     print("-" * 50)
    #     print(comp)
    #     print("-" * 50)
    #     print(retrieval)
    #     print("-" * 50)
    #     print(origin_tokens)
    #     print("-" * 50)
    #     print(comp_tokens)
    #     print("-" * 50)
    #     print(retrieval_tokens)
    #     print("=" * 50)

    #     print(
    #         f"comp rate: {comp_rate}, variation_rate: {variation_rate}, alignment_gap: {alignment_gap}"
    #     )

    res[chunk_idx] = {
        "labels": labels,
        "origin": origin,
        "comp": comp,
        "retrieval": retrieval,
        "origin_tokens": origin_tokens,
        "comp_rate": comp_rate,
        "variation_rate": variation_rate,
        "hitting_rate": hitting_rate,
        "matching_rate": matching_rate,
        "alignment_gap": alignment_gap,
    }

    res_pt["labels"].append(labels)
    res_pt["origin"].append(origin)
    res_pt["comp"].append(comp)
    res_pt["retrieval"].append(retrieval)
    res_pt["origin_tokens"].append(origin_tokens)
    res_pt["comp_rate"].append(comp_rate)
    res_pt["variation_rate"].append(variation_rate)
    res_pt["hitting_rate"].append(hitting_rate)
    res_pt["matching_rate"].append(matching_rate)
    res_pt["alignment_gap"].append(alignment_gap)

    # if int(chunk_idx) % 1000 == 0:
    #     json.dump(res, open(save_path, "w"), indent=4)
    #     torch.save(res_pt, save_path.replace(".json", ".pt"))

# save_path = 'output/longllmlingua/nq10_label_word_llama2.json'
save_path = f'output/longllmlingua/{dataset_name}_label_word{llama2}.json'
# save_path = 'output/longllmlingua/hotpotqa_label_word.json'
# save_path = 'output/longllmlingua/2wiki_label_word.json'

json.dump(res, open(save_path, "w"), indent=4)
torch.save(res_pt, save_path.replace(".json", ".pt"))

compression_rate_avg = compression_rate_avg / num_sample
find_rate_avg = find_rate_avg / num_sample
variation_rate_avg = variation_rate_avg / num_sample
matching_rate_avg = matching_rate_avg / num_sample
hitting_rate_avg = hitting_rate_avg / num_sample
alignment_gap_avg = alignment_gap_avg / num_sample

print_info = f"window size: {window_size}, comp rate: {compression_rate_avg}, hitting_rate: {hitting_rate_avg}, retrieval rate: {matching_rate_avg}"
print(print_info)


import argparse
from collections import defaultdict

import numpy as np
import torch

## filtering
variation_rate_list = res_pt["variation_rate"]
print(len(variation_rate_list))
threshold = np.percentile(variation_rate_list, 90)
kept, filtered = defaultdict(list), defaultdict(list)
for labels, origin, comp, retrieval, cr, vr, hr, mr, ag in zip(
    res_pt["labels"],
    res_pt["origin"],
    res_pt["comp"],
    res_pt["retrieval"],
    res_pt["comp_rate"],
    res_pt["variation_rate"],
    res_pt["hitting_rate"],
    res_pt["matching_rate"],
    res_pt["alignment_gap"],
):
    if vr >= threshold:
        filtered["labels"].append(labels)
        filtered["origin"].append(origin)
        filtered["comp"].append(comp)
        filtered["retrieval"].append(retrieval)
        filtered["comp_rate"].append(cr)
        filtered["variation_rate"].append(vr)
        filtered["hitting_rate"].append(hr)
        filtered["matching_rate"].append(mr)
        filtered["alignment_gap"].append(ag)
    else:
        kept["labels"].append(labels)
        kept["origin"].append(origin)
        kept["comp"].append(comp)
        kept["retrieval"].append(retrieval)
        kept["comp_rate"].append(cr)
        kept["variation_rate"].append(vr)
        kept["hitting_rate"].append(hr)
        kept["matching_rate"].append(mr)
        kept["alignment_gap"].append(ag)

alignment_gap_list = kept["alignment_gap"]
threshold = np.percentile(alignment_gap_list, 90)
kept2 = defaultdict(list)
for labels, origin, comp, retrieval, cr, vr, hr, mr, ag in zip(
    kept["labels"],
    kept["origin"],
    kept["comp"],
    res_pt["retrieval"],
    kept["comp_rate"],
    kept["variation_rate"],
    kept["hitting_rate"],
    kept["matching_rate"],
    kept["alignment_gap"],
):
    if ag >= threshold:
        filtered["labels"].append(labels)
        filtered["origin"].append(origin)
        filtered["comp"].append(comp)
        filtered["retrieval"].append(retrieval)
        filtered["comp_rate"].append(cr)
        filtered["variation_rate"].append(vr)
        filtered["hitting_rate"].append(hr)
        filtered["matching_rate"].append(mr)
        filtered["alignment_gap"].append(ag)
    else:
        kept2["labels"].append(labels)
        kept2["origin"].append(origin)
        kept2["comp"].append(comp)
        kept2["retrieval"].append(retrieval)
        kept2["comp_rate"].append(cr)
        kept2["variation_rate"].append(vr)
        kept2["hitting_rate"].append(hr)
        kept2["matching_rate"].append(mr)
        kept2["alignment_gap"].append(ag)

torch.save(kept2, save_path)
print(save_path)
