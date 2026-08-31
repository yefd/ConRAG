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


# nohup python longllmlingua_data_pre.py >  logs/llmlingua/output_longllmlingua_nq10_mistral.log 2>&1 &
# nohup python longllmlingua_data_pre.py >  logs/llmlingua/output_longllmlingua_nq20_mistral.log 2>&1 &
# nohup python longllmlingua_data_pre.py >  logs/llmlingua/output_longllmlingua_hotpotqa_mistral.log 2>&1 &
# nohup python longllmlingua_data_pre.py >  logs/llmlingua/output_longllmlingua_musique_mistral.log 2>&1 &
# nohup python longllmlingua_data_pre.py >  logs/llmlingua/output_longllmlingua_2wiki_mistral.log 2>&1 &
# nohup python longllmlingua_data_pre.py >  logs/llmlingua/output_longllmlingua_mrag_mistral.log 2>&1 &


dataset_name = 'mrag'
# llama2 = ''
llama2 = '_llama2'

# Setup LLMLingua
from baselines.LLMLingua.llmlingua.prompt_compressor import PromptCompressor

llm_lingua = PromptCompressor(
    model_name='/Llama-2-7b-chat-hf'
    )

from transformers import AutoTokenizer

model_name = '/Mistral-7B-Instruct-v0.3' if llama2 == '' else '/Llama-2-7b-chat-hf'
save_path = f'output/longllmlingua/longllmlingua_compress_results_{dataset_name}{llama2}.pkl'
print('save_path', save_path)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
from tqdm import tqdm
from copy import deepcopy
def get_compressed_prompt(example):
    c = example['instruction'].split("\n\n")
    instruction, question = c[0], c[-1]
    demonstration = "\n".join(c[1:-1])
    # params are copied from https://github.com/microsoft/LLMLingua/blob/main/Transparency_FAQ.md#how-to-reproduce-the-result-in-llmlingua--longllmlingua
    compressed_prompt = llm_lingua.compress_prompt(
        demonstration.split("\n"),
        instruction,
        question,
        0.55,
        use_sentence_level_filter=False,
        condition_in_question="after_condition",
        reorder_context="sort",
        dynamic_context_compression_ratio=0.3, # or 0.4
        condition_compare=True,
        context_budget="+100",
        rank_method="longllmlingua",
    )
    new_example = deepcopy(example)
    new_example['compressed_prompt'] = compressed_prompt
    return new_example

if dataset_name == 'nq10':
    from ConRAG.dataset.load_nq import *
    instruction_dataset_train, instruction_dataset_test = load_nq_dataset(
        'noise/ConRAG/dataset/nq/nq-open-10_total_documents_gold_bert.pkl', 
        4096,
        tokenizer,
        retrieval_aware=False,
        use_cot=False, 
        add_noise=False, 
        noise_type='', 
        gt_position='none',
        use_emb=False,
    )

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
        ignore_train=True,
        )

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
        ignore_train=True,
        )

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
        max_prompt_length=4096, tokenizer=tokenizer, ignore_train=True)



print('len', len(instruction_dataset_test))

instruction_dataset_test = [get_compressed_prompt(example) for example in tqdm(instruction_dataset_test)]
print('save_path', save_path)
pickle.dump(instruction_dataset_test, open(save_path, 'wb'))






