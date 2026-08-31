import torch
import numpy as np
import random, os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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

import os
import re 
import time 
import pickle
import argparse
import numpy as np
from copy import deepcopy 
from tqdm import trange, tqdm
from typing import Union, Optional, Tuple, List, Dict

import torch 
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel 
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
import sys
sys.path.append('baselines/trace/')
from prompts import generate_knowledge_triples_template, generate_knowledge_triples_chat_template
from utils.utils import * 
from utils.const import HF_TOKEN

from generate_knowledge_triples import get_dataset_demonstrations, parse_model_output, add_sentence_index_to_generated_triples

from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

def generate_from_vllm(texts, vllm_model):
    prompts_input = [data for data in texts]
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100, skip_special_tokens=False)
    raw_responses = vllm_model.generate(prompts_input[:], sampling_params)
    responses = [output.outputs[0].text.strip('<|start_header_id|>assistant<|end_header_id|>').strip() for output in raw_responses]
    return responses


def get_document_prompts_chat_format_for_instruction_model(max_length, dataset_name, num_examplars, document_list: List[Dict[str, Optional[Union[str, List[str], List[int]]]]], tokenizer=None):

    # document_list: [{"title": str, "sentences": [str], "ranked_prompt_indices": [int] / None}]

    def convert_several_examplars_to_text(examplars):
        return "\n\n".join(examplars)

    def vary_num_examplars_based_on_context_window(examplars, title, text, tokenizer,):

        final_examplars = None

        while len(examplars) > 0:
            for num in range(len(examplars), 0, -1):
                possible_prompt = generate_knowledge_triples_template.format(
                    examplars=convert_several_examplars_to_text(examplars[:num]),
                    title=title, 
                    text=text, 
                )
                possible_prompt_tokens = tokenizer.encode(possible_prompt)
                if len(possible_prompt_tokens) <= max_length:
                    final_examplars = examplars[:num]
                    break
            if final_examplars is None:
                examplars = examplars[1:]
            else:
                break
        if final_examplars is None:
            final_examplars = []
        return final_examplars

    prompts = []
    for i, document in enumerate(document_list):

        title = document["title"]
        sentences = document["sentences"]
        text = " ".join(sentences)

        dataset_demonstrations = get_dataset_demonstrations(dataset_name)
        ranked_prompt_indices = document.get("ranked_prompt_indices", None)
        if ranked_prompt_indices is None:
            ranked_prompt_indices = np.random.permutation(len(dataset_demonstrations))

        examplars = [dataset_demonstrations[idx] for idx in ranked_prompt_indices[:num_examplars]]
        examplars = ["Title: {}\nText: {}\nKnowledge Triples: {}".format(example["title"], example["text"], example["triples"]) for example in examplars]
        examplars = vary_num_examplars_based_on_context_window(examplars, title, text, tokenizer)
        # LLaMA3 Format 
        prompts.append(
            tokenizer.apply_chat_template([
                {"role": "system", "content": generate_knowledge_triples_chat_template.format(examplars=convert_several_examplars_to_text(examplars))},
                {"role": "user", "content": f"Title: {title}\nText: {text}\nKnowledge Triples: "},
            ], tokenize=False, add_generation_prompt=True)
        )
            
    return prompts

def tokenizer_encode_chat_format_for_instruction_model(prompts: List[List[dict]], max_length: int=4096, tokenizer=None) -> Dict[str, Tensor]:
    
    texts = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
    batch_dict = tokenizer(texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
    tokenizer_outputs = {"input_ids": batch_dict["input_ids"], "attention_mask": batch_dict["attention_mask"]}

    return tokenizer_outputs

# def model_generate_kg(inputs: Dict[str, Tensor], max_new_tokens: int=100, model=None) -> Tensor:
#     device = 'cuda'
#     model = get_model() if model is None else model 
#     inputs = to_device(inputs, device)
#     generated_token_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True, temperature=1.0, do_sample=False) 
#     return generated_token_ids 

def generate_triples_for_document_list(batch_size, max_length, max_new_tokens, dataset_name, num_examplars, document_list: List[Dict[str, Optional[Union[str, List[str], List[int]]]]], verbose=False, tokenizer=None, model=None) -> List[List[Dict[str, Union[str, list]]]]:

    # if verbose:
    #     progress_bar = trange((len(document_list)-1) // batch_size + 1, desc="Generating Knowledge Triples")

    generated_contents = []
    batch_prompts = get_document_prompts_chat_format_for_instruction_model(max_length=max_length, dataset_name=dataset_name, num_examplars=num_examplars, document_list=document_list, tokenizer=tokenizer)
    generated_contents = generate_from_vllm(batch_prompts, model)
    # for i in range((len(document_list)-1) // batch_size + 1): 
    #     batch_document_list = document_list[i*batch_size: (i+1)*batch_size]
    #     batch_prompts = 
        # batch_inputs = tokenizer_encode_chat_format_for_instruction_model(prompts=batch_prompts, max_length=max_length, tokenizer=tokenizer)
        # batch_generated_token_ids = model_generate_kg(batch_inputs, max_new_tokens, model)
        # batch_generated_token_ids = batch_generated_token_ids[:, batch_inputs["input_ids"].shape[1]:]
        # batch_generated_texts = tokenizer.batch_decode(batch_generated_token_ids, skip_special_tokens=True)
        # batch_generated_texts = generate_from_vllm(batch_prompts, model)
        # generated_contents.extend(batch_generated_texts)
        # if verbose:
        #     progress_bar.update(1)
    
    # parse model_outputs 
    results = [] 
    for j, (document, generated_content) in enumerate(zip(document_list, generated_contents)):
        triples = parse_model_output(generated_content) # [(head, relation, tail)]
        document_text = " ".join(document["sentences"])
        triples_in_one_document = []
        for head, relation, tail in triples:
            if head.lower() != document["title"].lower():
                if head.lower() not in document_text.lower():
                    head = document["title"]
            
            triples_in_one_document.append(
                {
                    "head": head,
                    "relation": relation, 
                    "tail": tail, 
                }
            )
        results.append(triples_in_one_document)

    return results

def generate_document_knowledge_graph(dataset, dataset_name='hotpotqa', tokenizer=None, model=None):
    
    # obtain unique documents from the input data
    print('generate_document_knowledge_graph, len(dataset):', len(dataset))
    max_length = 4096
    num_examplars = 3
    batch_size = 1
    max_new_tokens = 512
    dataset_name = dataset_name
    id2document = {}
    for example in tqdm(dataset, desc="Obtaining Unique documents", total=len(dataset)):
        for ctx in example["ctxs_kg"]:
            ctx_text = f"(Title: {ctx['title']}) {ctx['text']}"
            document_hash_id = hash_object(ctx_text)
            if document_hash_id in id2document:
                assert ctx["title"] == id2document[document_hash_id]["title"] # the title must be the same 
                continue
            id2document[document_hash_id] = {
                "title": ctx["title"], 
                "sentences": ctx["sentences"],
                "ranked_prompt_indices": ctx.get("ranked_prompt_indices", None), 
            }
    print(f"Successfully get {len(id2document)} unique documents!")

    # obtain KG triples for unique documents 
    document_id_list = list(id2document.keys())
    document_id_to_index = {document_id: i for i, document_id in enumerate(document_id_list)}
    document_list = [id2document[document_id] for document_id in document_id_list]

    # tmp_document_triples_file = os.path.join(
    #     os.path.dirname(save_data_file), 
    #     "tmp_"+os.path.basename(save_data_file).split(".")[0]+".pkl"
    # )
    # if os.path.exists(tmp_document_triples_file):
    #     print(f"{tmp_document_triples_file} already exists, loading cached file...")
    #     document_triples_list = pickle.load(open(tmp_document_triples_file, "rb"))
    # else:
    document_triples_list = generate_triples_for_document_list(batch_size=batch_size, max_length=max_length, max_new_tokens=max_new_tokens, dataset_name=dataset_name, num_examplars=num_examplars, document_list=document_list, verbose=True, tokenizer=tokenizer, model=model)
        # pickle.dump(document_triples_list, open(tmp_document_triples_file, "wb"))

    torch.cuda.empty_cache()
    # obtain sentence index
    document_triples_with_sentence_index_list = [] 
    batch_size = 20000
    for i in trange((len(document_list) - 1) // batch_size + 1, desc="Finding triple sentence index"):
        batch_document_list = document_list[i*batch_size: (i+1)*batch_size]
        batch_document_triples_list = document_triples_list[i*batch_size: (i+1)*batch_size]
        document_triples_with_sentence_index_list.extend(
            add_sentence_index_to_generated_triples(batch_document_list, batch_document_triples_list)
        )

    # obtain document index 
    for example in tqdm(dataset, desc="Obtaining document triples", total=len(dataset)):
        for i, ctx in enumerate(example["ctxs_kg"]):
            ctx_text = f"(Title: {ctx['title']}) {ctx['text']}"
            document_hash_id = hash_object(ctx_text)
            document_triples = document_triples_with_sentence_index_list[document_id_to_index[document_hash_id]] # [{"head": str, "relation": str, "tail": str, "position": [None, sentence_index]}]
            document_triples = deepcopy(document_triples)
            for triple in document_triples:
                triple["position"][0] = i 
            ctx["triples"] = document_triples
    return dataset
    # save_json(data, save_data_file, use_indent=True)
    # print(f"Successfully save data to {save_data_file} ...")

    # if os.path.exists(tmp_document_triples_file):
    #     os.remove(tmp_document_triples_file)

import nltk 
from nltk.tokenize import sent_tokenize

def convert_musique_to_uniform_format(example):
    ctxs = []
    example['answers'] = [example['answer']]+example['answer_aliases']
    for i, context_item in enumerate(example["paragraphs"]):
        sentences = sent_tokenize(context_item["paragraph_text"])
        ctxs.append(
            {
                "id": str(i), 
                "title": context_item["title"], 
                "text": context_item["paragraph_text"], 
                "sentences": sentences, 
            }
        )
    example['ctxs_kg'] = ctxs
    return example

def convert_nq_to_uniform_format(example):
    ctxs = []
    for i, context_item in enumerate(example["ctxs"]):
        sentences = sent_tokenize(context_item["text"])
        ctxs.append(
            {
                "id": str(i), 
                "title": context_item["title"], 
                "text": context_item["text"], 
                "sentences": sentences, 
            }
        )
    example['ctxs_kg'] = ctxs
    return example


def convert_hotpotqa_to_uniform_format(example):
    ctxs = []
    example['answers'] = [example['answer']]
    for i, context_item in enumerate(example["context"]):
        sentences = sent_tokenize(context_item["text"])
        ctxs.append(
            {
                "id": str(i), 
                "title": context_item["title"], 
                "text": context_item["text"], 
                "sentences": sentences, 
            }
        )
    example['ctxs_kg'] = ctxs
    return example

def convert_mrag_to_uniform_format(example):
    ctxs = []
    example['answers'] = [example['answer']]
    example['question'] = example['query']
    for i, context_item in enumerate(example["ctxs"]):
        sentences = sent_tokenize(context_item["text"])
        ctxs.append(
            {
                "id": str(i), 
                "title": '', 
                "text": context_item["text"], 
                "sentences": sentences, 
            }
        )
    example['ctxs_kg'] = ctxs
    return example


    # elif dataset_name == 'bioasq':
    #     convert_func = get_bioasq_ans
    # elif dataset_name == 'lkqa':
    #     convert_func = get_lkqa_ans

import os
import time
import argparse
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from typing import List, Dict

import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging as hf_logging

from utils.utils import *
from utils.const import HF_TOKEN
from retrievers.e5_mistral import get_e5_mistral_embeddings_for_query, get_e5_mistral_embeddings_for_document


def get_dataset_demonstrations_reasoning(dataset_name: str):
    if dataset_name == "hotpotqa":
        from prompts import generate_reasoning_chains_hotpotqa_examplars, reasoning_chains_hotpotqa_examplars
        return generate_reasoning_chains_hotpotqa_examplars, reasoning_chains_hotpotqa_examplars
    elif dataset_name == "2wikimultihopqa":
        from prompts import generate_reasoning_chains_2wikimultihopqa_examplars, reasoning_chains_2wikimultihopqa_examplars
        return generate_reasoning_chains_2wikimultihopqa_examplars, reasoning_chains_2wikimultihopqa_examplars
    elif dataset_name == "musique":
        from prompts import generate_reasoning_chains_musique_examplars, reasoning_chains_musique_examplars
        return generate_reasoning_chains_musique_examplars, reasoning_chains_musique_examplars
    else:
        raise ValueError(f"{dataset_name} is not a supported dataset!")


def get_llama3_generate_reasoning_chains_prompts_chat_format(
    hop: int, 
    question: str, 
    existing_triples: List[List[str]], 
    candidate_triples: List[List[str]],
    max_length: int, 
    num_examplars: int,
    disable_demonstration: bool,
    ranked_prompt_indices: list = None, 
    tokenizer: AutoTokenizer = None,
    model=None,
    dataset_name='',
) -> List[str]:

    def convert_candidate_triples_to_choices(candidates): 
        return "\n".join(["A. no need for additional knowledge triples"] \
            + ["{}. {}".format(chr(ord('B')+k), triple) for k, triple in enumerate(candidates)])

    def convert_several_examplars_to_text(examplars):
        return "\n\n".join(examplars)
    
    def vary_num_examplars_based_on_context_window(instruction, examplars, question, triples, candidates, max_length):
        final_examplars = None
        while len(examplars) > 0:
            for num in range(len(examplars), 0, -1):
                possible_prompt = "{} {}\n\nquestion: {}\nknowledge triples: {}\ncandidate knowledge triples:\n{}\nanswer:".format(
                    instruction, convert_several_examplars_to_text(examplars[:num]), 
                    question, " ".join(triples), convert_candidate_triples_to_choices(candidates)
                )
                possible_prompt_tokens = tokenizer.encode(possible_prompt)
                if len(possible_prompt_tokens) <= max_length:
                    final_examplars = examplars[:num]
                    break
            if final_examplars is None:
                examplars = examplars[1:]
            else:
                break
        if final_examplars is None:
            final_examplars = [] 
        return final_examplars

    prompts = []
    for triples, candidates in zip(existing_triples, candidate_triples):
        instruction = "Select the next knowledge triple that extends an existing set of knowledge triples to form a coherent reasoning path capable of answering a specified question. " \
            "If the current reasoning path is sufficient to answer the question, simply output A. Please only output the choice for the next knowledge triple."
        
        if not disable_demonstration:
            instruction += "\n\nThe followings are some examples of coherent reasoning paths capable of answering the specified question " \
                f"and how the {hop+1}-th knowledge triples in these paths are selected:\n\n"
            generate_reasoning_chains_examplars, reasoning_chains_examplars = get_dataset_demonstrations_reasoning(dataset_name)  # Hardcoded for example
            if ranked_prompt_indices is not None:
                reasoning_chains_examplars = [reasoning_chains_examplars[idx] for idx in ranked_prompt_indices]
                generate_reasoning_chains_examplars = [generate_reasoning_chains_examplars[idx] for idx in ranked_prompt_indices]

            examplars = [] 
            for i, (rp_examplar, grp_examplar) in enumerate(
                zip(
                    reasoning_chains_examplars, 
                    generate_reasoning_chains_examplars
                )
            ):
                if len(grp_examplar) < hop + 1:
                    continue 
                examplar = "coherent reasoning path: {}\nquestion: {}\n".format(rp_examplar["chains"], rp_examplar["question"])
                examplar += "The {}-th triple in the reasoning path is selected as:\n".format(hop+1)
                one_step_item = grp_examplar[hop]
                examplar += "existing knowledge triples: {}\nquestion: {}\ncandidate knowledge triples:\n{}\nthe next possible triple is:{}\n".format(
                    ", ".join(one_step_item["triples"]), one_step_item["question"], "\n".join(one_step_item["candidate_triples"]), one_step_item["answer"]
                )
                examplars.append(examplar)
                if len(examplars) >= num_examplars:
                    break 
            examplars = vary_num_examplars_based_on_context_window(instruction, examplars, question, triples, candidates, max_length)
            instruction += convert_several_examplars_to_text(examplars)
        else:
            instruction += "\n\n"
        
        user_input_text = "The {}-th triple in the reasoning path is selected as:\nexisting knowledge triples: {}\nquestion: {}\ncandidate knowledge triples:\n{}\nthe next possible triple is:".format(
            hop+1, ", ".join(triples), question, convert_candidate_triples_to_choices(candidates)
        )
        prompts.append(
            [
                {"role": "system", "content": instruction}, 
                {"role": "user", "content": user_input_text}
            ]
        )
    
    return prompts

def model_generate(inputs: Dict[str, Tensor], max_new_tokens: int=100, batch_size=2, tokenizer=None, model=None) -> Tensor:
    device = 'cuda'
    generated_token_ids_list, generated_token_logits_list = [], [] 
    for i in range((len(inputs["input_ids"])-1)//batch_size+1):
        batch_inputs = {k: v[i*batch_size: (i+1)*batch_size] for k, v in inputs.items()}
        batch_inputs = to_device(batch_inputs, device)
        batch_outputs = model.generate(**batch_inputs, max_new_tokens=max_new_tokens, output_scores=True, return_dict_in_generate=True, do_sample=False, temperature=1.0) # temperature=1.5, do_sample=True) # 
        batch_generated_token_ids = batch_outputs.sequences[:, batch_inputs["input_ids"].shape[1]:].detach().cpu()
        batch_generated_token_logits = torch.cat([token_scores.unsqueeze(1) for token_scores in batch_outputs.scores], dim=1).detach().cpu()

        if batch_generated_token_ids.shape[1] < max_new_tokens:
            real_batch_size, num_generated_tokens = batch_generated_token_ids.shape 
            padding_length = max_new_tokens-num_generated_tokens
            padding_token_ids = torch.zeros((real_batch_size, padding_length), dtype=batch_generated_token_ids.dtype).fill_(tokenizer.pad_token_id)
            padding_token_logits = torch.zeros((real_batch_size, padding_length, batch_generated_token_logits.shape[-1]), dtype=batch_generated_token_logits.dtype)
            batch_generated_token_ids = torch.cat([batch_generated_token_ids, padding_token_ids], dim=1)
            batch_generated_token_logits = torch.cat([batch_generated_token_logits, padding_token_logits], dim=1)
        
        generated_token_ids_list.append(batch_generated_token_ids)
        generated_token_logits_list.append(batch_generated_token_logits)

    generated_token_ids = torch.cat(generated_token_ids_list, dim=0)
    generated_token_logits = torch.cat(generated_token_logits_list, dim=0)

    return generated_token_ids, generated_token_logits

token_id_to_choice_map = None
def get_answer_token_indices(num_choices, token_ids, tokenizer):

    global token_id_to_choice_map # glabal param is a shit
    if token_id_to_choice_map is None:
        token_id_to_choice_map = {}
        choices = [chr(ord('A')+i) for i in range(num_choices+1)] 
        for choice in choices:
            token_id_to_choice_map[tokenizer.encode(choice, add_special_tokens=False)[0]] = choice
            token_id_to_choice_map[tokenizer.encode(" {}".format(choice), add_special_tokens=False)[-1]] = choice
    
    answer_token_indices = torch.zeros((token_ids.shape[0],), dtype=token_ids.dtype).fill_(token_ids.shape[1]-1)
    for i in range(token_ids.shape[0]):
        for j in range(token_ids.shape[1]):
            if token_ids[i, j].item() in token_id_to_choice_map:
                answer_token_indices[i] = j 
                break

    return answer_token_indices

def construct_reasoning_chains(dataset_name, dataset, tokenizer, model):
    max_chain_length = 4
    num_choices = 20
    num_examplars = 3
    max_length = 2048
    max_new_tokens = 16
    num_beams = 5
    num_chains = 20
    min_triple_prob = 1e-4
    calculate_ranked_prompt_indices = True
    disable_demonstration=False
    device = 'cuda'
    
    if calculate_ranked_prompt_indices:
        _, reasoning_chains_examplars = get_dataset_demonstrations_reasoning(dataset_name)
        questions_in_examplars = [item["question"] for item in reasoning_chains_examplars]
        questions_in_data = [item["question"] for item in dataset]
        print("Calculating E5-Mistral Embeddings of Questions in Prompts ... ")
        questions_in_prompts_embeddings = get_e5_mistral_embeddings_for_document(questions_in_examplars, max_length=128, batch_size=2)
        print("Calculating E5-Mistral Embeddings of Questions in Data ... ")
        questions_in_data_embeddings = get_e5_mistral_embeddings_for_document(questions_in_data, max_length=128, batch_size=2)

        similarities = torch.matmul(questions_in_data_embeddings, questions_in_prompts_embeddings.T)
        ranked_prompt_indices = torch.argsort(similarities, dim=1, descending=True)
        for example, one_ranked_prompt_indices in zip(dataset, ranked_prompt_indices):
            example["ranked_prompt_indices"] = one_ranked_prompt_indices.tolist()

    global token_id_to_choice_map
    for example in tqdm(dataset, desc="Generating reasoning chains", total=len(dataset)):

        question = example["question"]
        triples, triple_positions = [], []
        for ctx in example["ctxs_kg"]:
            for triple_item in ctx["triples"]:
                triples.append("<{}; {}; {}>".format(triple_item["head"], triple_item["relation"], triple_item["tail"]))
                triple_positions.append(triple_item["position"])
        
        num_total_triples = len(triples)
        triples_embeddings = get_e5_mistral_embeddings_for_document(triples, max_length=128, batch_size=2)

        paths = [[]]
        paths_scores = [1.0]
        paths_finished = [False]
        for j in range(max_chain_length):
            if np.sum(paths_finished) == num_chains:
                break
            queries = [
                "knowledge triples: {}\nquestion: {}".format(
                    " ".join([triples[idx] for idx in path]), question
                )
                for path in paths
            ]
            queries_embeddings = get_e5_mistral_embeddings_for_query("retrieve_relevant_triples", queries, max_length=256, batch_size=1)

            queries_triples_similarities = torch.matmul(queries_embeddings, triples_embeddings.T) # n_path, n_triples 

            candidate_triples_mask = torch.ones_like(queries_triples_similarities)
            for k, path in enumerate(paths):
                candidate_triples_mask[k, path] = 0.0 
            queries_triples_similarities = queries_triples_similarities + \
                torch.finfo(queries_triples_similarities.dtype).min * (1.0 - candidate_triples_mask)
            topk_most_relevant_triples_indices = torch.topk(queries_triples_similarities, k=min(num_choices, num_total_triples), dim=1)[1].tolist()

            prompts = get_llama3_generate_reasoning_chains_prompts_chat_format(
                hop = j, 
                question = question,
                existing_triples = [[triples[idx] for idx in path] for path in paths],
                candidate_triples = [
                    [triples[idx] for idx in candidate_triples_indices] \
                        for candidate_triples_indices in topk_most_relevant_triples_indices
                ],
                ranked_prompt_indices = example.get("ranked_prompt_indices", None),
                max_length=max_length,
                num_examplars=num_examplars,
                disable_demonstration=disable_demonstration,
                tokenizer=tokenizer,
                dataset_name=dataset_name,
                model=model,
            )
            inputs = tokenizer_encode_chat_format_for_instruction_model(prompts, max_length, tokenizer=tokenizer)
            generated_token_ids, generated_token_logits = model_generate(inputs, max_new_tokens=max_new_tokens, batch_size=2, model=model, tokenizer=tokenizer)

            answer_token_indices = get_answer_token_indices(num_choices, generated_token_ids, tokenizer=tokenizer)
            answer_token_logits = generated_token_logits.gather(1, \
                answer_token_indices[:, None, None].expand(-1, -1, generated_token_logits.shape[-1]))
            answer_token_logits = answer_token_logits.squeeze(1)

            choices_token_ids_list = list(token_id_to_choice_map.keys())
            choices_list = [token_id_to_choice_map[token_id] for token_id in choices_token_ids_list]
            answer_token_probs = F.softmax(answer_token_logits[:, choices_token_ids_list], dim=1)

            new_paths, new_paths_scores, new_paths_finished = [], [], []
            topk_choices_probs, topk_choices_indices = torch.topk(answer_token_probs, k=num_beams, dim=1)

            for i in range(len(paths)):
                if paths_finished[i]:
                    new_paths.append(paths[i])
                    new_paths_scores.append(paths_scores[i])
                    new_paths_finished.append(True)
                    continue 
                if torch.all(torch.isnan(topk_choices_probs[i])):
                    print("No choice in generated results! generated text: {}".format(tokenizer.decode(generated_token_ids[i])))
                    new_paths.append(paths[i])
                    new_paths_scores.append(paths_scores[i])
                    new_paths_finished.append(False)
                    continue 
                for b in range(num_beams):
                    if torch.isnan(topk_choices_probs[i, b]) or topk_choices_probs[i, b].item() < min_triple_prob:
                        continue
                    current_choice = choices_list[topk_choices_indices[i, b].item()]
                    if current_choice != 'A' and (ord(current_choice)-ord('B') >= len(topk_most_relevant_triples_indices[i])):
                        continue
                    new_paths_scores.append(paths_scores[i]*topk_choices_probs[i, b].item())
                    if current_choice == 'A':
                        new_paths.append(paths[i]+[-1]) 
                        new_paths_finished.append(True)
                    else:
                        new_paths.append(paths[i]+[topk_most_relevant_triples_indices[i][ord(current_choice)-ord('B')]])
                        new_paths_finished.append(False)
            
            assert len(new_paths) == len(new_paths_scores)
            assert len(new_paths) == len(new_paths_finished)
            new_paths_sorted_indices = sorted(range(len(new_paths_scores)), key=lambda x: new_paths_scores[x], reverse=True)
            topk_new_paths_sorted_indices = new_paths_sorted_indices[:num_chains]
            paths = [new_paths[idx] for idx in topk_new_paths_sorted_indices]
            paths_scores = [new_paths_scores[idx] for idx in topk_new_paths_sorted_indices]
            paths_finished = [new_paths_finished[idx] for idx in topk_new_paths_sorted_indices]
        
        example["chains"] = [
            {
                "triples":[
                    {
                        "triple": triples[triple_index], 
                        "triple_position": triple_positions[triple_index]
                    } 
                    for triple_index in path if triple_index >= 0 
                ],
                "score": path_score
            }
            for path, path_score in zip(paths, paths_scores)
        ]
    return dataset
    # print(f"saving data to {save_data_file} ... ")
    # os.makedirs(os.path.dirname(save_data_file), exist_ok=True)
    # save_json(data, save_data_file, use_indent=True)

import os
import logging
import argparse
import numpy as np
from tqdm import tqdm 

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging as hf_logging

from readers.datasets import ReaderDatasetWithChains
from readers.collators import CollatorWithChainsChatFormat, CollatorWithChains
from readers.metrics import ems
from readers.metrics import f1_score as f1_score_fn

from utils.const import * 
from utils.utils import seed_everything, setup_logger, to_device

from evaluation import parse_generated_answer, parse_generated_answer_chat_format

def evaluate(tokenizer, dataloader, model, answer_maxlength):

    em_scores_list, f1_scores_list, precision_scores_list, recall_scores_list, num_tokens_list = [], [], [], [], []
    results = []
    for batch_index, batch_inputs in tqdm(dataloader, desc="Evaluation", total=len(dataloader)):
        batch_inputs = to_device(batch_inputs, DEVICE)
        batch_outputs = model.generate(**batch_inputs, max_new_tokens=answer_maxlength, do_sample=False, temperature=1.0)
        batch_generated_token_ids = batch_outputs[:, batch_inputs["input_ids"].shape[1]:].detach().cpu()
        for i, o in enumerate(batch_generated_token_ids):
            ans = tokenizer.decode(o, skip_special_tokens=True)
            ans = parse_generated_answer(ans) # parse_generated_answer_chat_format
            gold = dataloader.dataset.get_example(batch_index[i])["answers"]
            em_scores_list.append(ems(ans, gold))
            # if not ems(ans, gold):
            #     print(ans, "\t", gold)
            f1, precision, recall = f1_score_fn(ans, gold[0])
            f1_scores_list.append(f1)
            precision_scores_list.append(precision)
            recall_scores_list.append(recall)
            num_tokens_list.append(batch_inputs["attention_mask"][i].sum().item())
            results.append(ans)

    metrics = {}
    metrics["exact_match"] = np.mean(em_scores_list)
    metrics["f1"] = np.mean(f1_scores_list)
    metrics["precision"] = np.mean(precision_scores_list)
    metrics["recall"] = np.mean(recall_scores_list)
    metrics["avg_num_tokens"] = np.mean(num_tokens_list)

    return metrics, results

def eval(tokenizer, model, dataset_name, dataset):
    text_maxlength = 4096
    answer_maxlength = 100

    n_context = 5
    context_type = 'triples'
    batch_size = 1
    seed = 42
    seed_it(seed)

    # load dataset  
    dataset = ReaderDatasetWithChains(dataset=dataset, n_context=n_context, chain_key="chains")

    # load collator  # CollatorWithChains
    COLLATOR_FN = CollatorWithChainsChatFormat if 'chat_template' in tokenizer.__dict__['init_kwargs'].keys() else CollatorWithChains
    collator = COLLATOR_FN(
        tokenizer=tokenizer, 
        text_maxlength=text_maxlength,
        answer_maxlength=answer_maxlength, 
        context_type=context_type 
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False, collate_fn=collator)

    # evaluate
    # for idx, batch in dataloader:
    #     print(batch)
    metrics, results = evaluate(tokenizer, dataloader, model=model, answer_maxlength=answer_maxlength)
    print(metrics)
    print("====================== Evaluation Results ======================")
    print("n_context: {}".format(n_context))
    print("context_type: {}".format(context_type))
    print(metrics)
    print("================================================================")
    return results

from ConRAG.utils.metrics import *
from runner import *

class TraceRunner(ConRAGRunner):
    def __init__(
        self,
        reasoning_model_name='Llama-3.1-8B-Instruct',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.reasoning_model_name = reasoning_model_name
        self.use_emb = False

    def load_vllm_model(self, ):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_name = self.reasoning_model_name

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        vllm_model = LLM(
                model=model_name,
                tensor_parallel_size=1,
                trust_remote_code=True,
                # download_dir=hf_cache_path,
                gpu_memory_utilization=0.9,
                load_format="auto",
                # max_num_batched_tokens=32768,
            )
        self.reasoning_tokenizer = tokenizer
        self.reasoning_model = vllm_model

    def load_reasoning_model(self,):
        tokenizer = AutoTokenizer.from_pretrained(self.reasoning_model_name, use_auth_token=True)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(self.reasoning_model_name, trust_remote_code=True, device_map="cuda", torch_dtype=torch.bfloat16)
        self.reasoning_tokenizer = tokenizer
        self.reasoning_model = model
    
    def convert_dataset_to_uniform_format(self, dataset_name, dataset):
        if 'nq' in dataset_name:
            convert_func = convert_nq_to_uniform_format
            reasoning_name = '2wikimultihopqa'

        elif dataset_name == 'hotpotqa':
            convert_func = convert_hotpotqa_to_uniform_format
            reasoning_name = 'hotpotqa'

        elif dataset_name == '2wiki':
            convert_func = convert_hotpotqa_to_uniform_format
            reasoning_name = '2wikimultihopqa'

        elif dataset_name == 'musique':
            convert_func = convert_musique_to_uniform_format
            reasoning_name = 'musique'

        elif dataset_name == 'mrag':
            convert_func = convert_mrag_to_uniform_format
            reasoning_name = '2wikimultihopqa'
        else:
            raise ValueError(dataset_name)
        print('reasoning name', reasoning_name)
        self.reasoning_name = reasoning_name
        return [convert_func(d) for d in dataset]
    
    def generate_kg(self):
        self.load_vllm_model()
        dataset_test = self.convert_dataset_to_uniform_format(self.dataset_name, self.instruction_dataset_test)
        dataset_kg = generate_document_knowledge_graph(dataset_test[:], self.reasoning_name, self.reasoning_tokenizer, self.reasoning_model)
        self.dataset_kg = dataset_kg
    
    def construct_reasoning_chains(self):
        import gc
        del self.reasoning_model
        gc.collect()
        torch.cuda.empty_cache()
        self.load_reasoning_model()
        dataset_reasoning = construct_reasoning_chains(dataset_name=self.reasoning_name, dataset=self.dataset_kg, model=self.reasoning_model, tokenizer=self.reasoning_tokenizer)
        self.dataset_reasoning = dataset_reasoning
    
    def generate_answers(self):
        import gc
        del self.reasoning_model
        import importlib
        from retrievers import e5_mistral
        importlib.reload(e5_mistral)
        gc.collect()
        torch.cuda.empty_cache()

        self.load_model()
        self.load_tokenizer()
        res = eval(tokenizer=self.tokenizer, model=self.model.llama_model, dataset_name=self.reasoning_name, dataset=self.dataset_reasoning)

        if self.save_results:
            save_pkl_file = f'res_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if not os.path.exists(f'output/{self.dataset_name}'):
                os.makedirs(f'output/{self.dataset_name}')
            pkl_save_path = f'output/{self.dataset_name}/trace_{save_pkl_file}.pkl'
            print('results_save_path', pkl_save_path)
            with open(pkl_save_path, 'wb') as f:
                pickle.dump(res, f)
                f.close()
        if 'nq' in self.dataset_name:
            get_ans = get_nq_ans
        elif self.dataset_name == 'hotpotqa' or self.dataset_name == '2wiki':
            get_ans = get_hotpotqa_ans
        elif self.dataset_name == 'musique':
            get_ans = get_musique_ans
        elif self.dataset_name == 'dureader':
            get_ans = get_dureader_ans
        elif self.dataset_name == 'mrag':
            get_ans = get_mrag_ans
        elif self.dataset_name == 'bioasq':
            get_ans = get_bioasq_ans
        elif self.dataset_name == 'lkqa':
            get_ans = get_lkqa_ans
        gt_ans = get_ans(self.instruction_dataset_test)
        m = evaluation_from_list(res[:], gt_ans[:len(res)], self.dataset_name)
    
    def run(self):
        self.load_dataset(ignore_train=True)
        self.generate_kg()
        self.construct_reasoning_chains()
        self.generate_answers()


def main(dataset_name, input_path, train_data_path, test_data_path, dataset_names, paths, **args):
    if dataset_name in ['hotpotqa', 'musique', '2wiki', 'bioasq']:
        input_path = {'train_data_path': train_data_path, 'test_data_path': test_data_path}
    print(args)
    runner = TraceRunner(
        dataset_name=dataset_name,
        input_path=input_path, 
        **args
        )
    print('runner', runner.__dict__)
    runner.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model with given parameters")
    
    parser.add_argument('--dataset_name', type=str, default='', choices=['nq_10', 'nq_20', 'nq_30', 'hotpotqa', 'musique', '2wiki', 'dureader', 'mrag', 'bioasq', 'lkqa'], help='Name of the dataset')
    parser.add_argument('--input_path', type=str, default=None, help='Path for nq or dureader datasets')
    parser.add_argument('--train_data_path', type=str, default=None, help='Path for the hotpotqa, musique, 2wiki')
    parser.add_argument('--test_data_path', type=str, default=None, help='Path for the hotpotqa, musique, 2wiki')
    parser.add_argument('--dataset_names', nargs='+',type=str, default=[], help='Name of the dataset')
    parser.add_argument('--paths', nargs='+', type=str, default=[], help='Paths for datasets')

    parser.add_argument('--max_prompt_length', type=int, default=4096, help='Maximum prompt length')
    parser.add_argument('--model_name', type=str, required=True, help='Name of LLM')
    parser.add_argument('--reasoning_model_name', type=str, required=True, help='Name of reasoning LLM')

    parser.add_argument('--freeze_llm', action='store_true', help='Freeze LLM')
    parser.add_argument('--input_dim', type=int, default=3, help='Input features')

    parser.add_argument('--load_in_8bit', action='store_true', help='Load in 8-bit precision')
    parser.add_argument('--use_flash_att', action='store_true', help='Load in use_flash_att')
    parser.add_argument('--model_type', type=str, default='rag', choices=['rag', 'conrag'], help='model type')
    parser.add_argument('--num_k', type=int, default=10, help='Number of retrieved documents')
    parser.add_argument('--use_evaluation', action='store_true', help='Use for evaluation')

    parser.add_argument('--save_results', action='store_true', help='Save results')
    parser.add_argument('--instruction_type', default='llama', choices=['chat', 'instruction', 'qwen'], help='instruction_type, llama or mistral')

    args = parser.parse_args()
    main(**vars(args))








