"""
Copy and revise from https://github.com/carriex/recomp/blob/main/train_extractive_compressor.py. Below is origin statement.
Adapted from : https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_bi-encoder_mnrl.py
This examples show how to train a Bi-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

For training, we use MultipleNegativesRankingLoss. There, we pass triplets in the format:
(query, positive_passage, negative_passage)

Negative passage are hard negative examples, that were mined using different dense embedding methods and lexical search methods.
Each positive and negative passage comes with a score from a Cross-Encoder. This allows denoising, i.e. removing false negative
passages that are actually relevant for the query.

With a distilbert-base-uncased model, it should achieve a performance of about 33.79 MRR@10 on the MSMARCO Passages Dev-Corpus

Running this script:
"""
import sys
import json
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, evaluation, losses, InputExample
import logging
from datetime import datetime
import gzip
import os
import tarfile
from collections import defaultdict
from torch.utils.data import IterableDataset
import tqdm
from torch.utils.data import Dataset
import random
import pickle
import argparse
import pandas as pd
from tqdm import tqdm

# copy from https://github.com/carriex/recomp/blob/main/run_extractive_compressor.py
def get_contriever_scores(query, ctxs, model):
    embeddings = model.encode([query] + ctxs, convert_to_tensor=True)
    q_emb = embeddings[0]
    c_emb = embeddings[1:]
    scores = []
    for idx in range(c_emb.shape[0]):
        scores.append(float(q_emb @ c_emb[idx]))
    return scores

class RALMDataset(Dataset):
    def __init__(self, dataset_name, dataset, model, num_negatives, batched=False):
        self.dataset = dataset
        self.triplet_data = []  # store dataset
        for data in tqdm(self.dataset):  # for each sample
            label = data['label']
            if 'nq' in dataset_name:
                save_key = 'ctxs'
                ctxs = ['Title: '+ctx['title'] +'\n' + ctx['text'] for ctx in data[save_key]]
                query = data['question']
            ### HotpotQA ### ### 2Wiki ###
            elif dataset_name == 'hotpotqa' or dataset_name == '2wiki':
                save_key = 'context'
                ctxs = ['Title: '+ctx['title'] +'\n' + ctx['text'] for ctx in data[save_key]]
                query = data['question']
            ### MuSiQue ###
            elif dataset_name == 'musique':
                save_key = 'paragraphs'
                ctxs = ['Title: '+ctx['title'] +'\n' + ctx['paragraph_text'] for ctx in data[save_key]]
                query = data['question']
            ### MRAG ###
            elif dataset_name == 'mrag':   
                save_key = 'ctxs'
                ctxs = [ctx['text'] for ctx in data[save_key]]
                query = data['query']
            contriever_scores = get_contriever_scores(query, ctxs, model)

            pos_idx = [index for index, value in enumerate(label) if value == 1]
            if len(pos_idx) == 0:
                continue
            neg_idx = [index for index, value in enumerate(label) if value == 0]
            for pos_ctx_idx in pos_idx:  # for each positive idx
                if pos_ctx_idx < len(ctxs):  # make sure pos in candidate documents

                    pos_ctx = ctxs[pos_ctx_idx]  # get the positive document
                    num_negative_ctxs = 0
                    for neg_ctx_idx in neg_idx:  # for each negative idx
                        neg_ctx = ctxs[neg_ctx_idx]  # get negative document
                        # if neg_ctx['em'] < pos_ctx['em']:               # check the negative document is REAL passive for LLM
                        if num_negative_ctxs < num_negatives:
                            if contriever_scores[neg_ctx_idx] > contriever_scores[pos_ctx_idx]:  # check the negative document is REAL passive for contriever
                                self.triplet_data.append(
                                    (query, pos_ctx, neg_ctx)
                                    )
                                num_negative_ctxs += 1

        print("Total training pairs: {}".format(len(self.triplet_data)))

    def __getitem__(self, item):
        query_text, pos_text, neg_text = self.triplet_data[item]
        return InputExample(texts=[query_text, pos_text, neg_text])

    def __len__(self):
        return len(self.triplet_data)

from copy import deepcopy
import pathlib

PROMPTS_ROOT = pathlib.Path('ConRAG/prompts').resolve()

def get_compress_instruction( dataset_name,
    question: str, documents, tokenizer, num_k):
    if not question:
        raise ValueError(f"Provided `question` must be truthy, got: {question}")
    if not documents:
        raise ValueError(f"Provided `documents` must be truthy, got: {documents}")

    prompt_filename = "recomp_single_qa.prompt" # if 'nq' in dataset_name else 'recomp_multi_qa.prompt'

    with open(PROMPTS_ROOT / prompt_filename) as f:
        prompt_template = f.read().rstrip("\n")

    formatted_documents = []
    for document_index, document in enumerate(documents[:num_k]):
        document_prompt = f"[{document_index+1}]{document}"
        formatted_documents.append(document_prompt)

    return prompt_template.format(question=question, search_results="\n".join(formatted_documents))

def get_extractive_results(dataset_name, dataset, model, num_k=5):
    new_dataset = []
    for data in tqdm(dataset):  # for each sample
        data = deepcopy(data)
        
        label = data['label']

        if 'nq' in dataset_name:
            save_key = 'ctxs'
            ctxs = ['Title: '+ctx['title'] +'\n' + ctx['text'] for ctx in data[save_key]]
            query = data['question']
        ### HotpotQA ### ### 2Wiki ###
        elif dataset_name == 'hotpotqa' or dataset_name == '2wiki':
            save_key = 'context'
            ctxs = ['Title: '+ctx['title'] +'\n' + ctx['text'] for ctx in data[save_key]]
            query = data['question']
        ### MuSiQue ###
        elif dataset_name == 'musique':
            save_key = 'paragraphs'
            ctxs = ['Title: '+ctx['title'] +'\n' + ctx['paragraph_text'] for ctx in data[save_key]]
            query = data['question']
        ### MRAG ###
        elif dataset_name == 'mrag':
            save_key = 'ctxs'
            ctxs = [ctx['text'] for ctx in data[save_key]]
            query = data['query']
        contriever_scores = get_contriever_scores(query, ctxs, model)
        rank_list = np.array(contriever_scores).argsort().tolist()[::-1]
        ctxs_text = [ctxs[i] for i in rank_list]

        data['contriever_scores'] = [contriever_scores[i] for i in rank_list]
        data[save_key] = [data[save_key][i] for i in rank_list]
        data['label'] = [label[i] for i in rank_list]
        data['compress_instruction'] = get_compress_instruction(dataset_name, query, ctxs_text, tokenizer, num_k)
        new_dataset.append(data)
    return new_dataset

import pickle
from transformers import AutoTokenizer
from sentence_transformers.losses import MultipleNegativesRankingLoss
import math
import numpy as np
model_name = 'Mistral-7B-Instruct-v0.3'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

def main(dataset_name, dataset_save_path, input_path=None, train_data_path=None, test_data_path=None, num_k=5):
    if dataset_name in ['hotpotqa', 'musique', '2wiki', 'bioasq']:
        input_path = {'train_data_path': train_data_path, 'test_data_path': test_data_path}
    train_batch_size = 64 #Increasing the train batch size improves the model performance, but requires more GPU memory
    max_seq_length = 512  # default is 100; #Max length for passages. Increasing it, requires more GPU memory
    train_data_path = ''
    dev_data_path = ''
    dev_data_num_tokens = -1
    model_name = 'contriever'
    max_passages = 10
    epochs = 3 # Number of epochs we want to train # cp from paper 
    pooling = "mean"
    warmup_steps = 1000
    lr = 2e-5
    use_pre_trained_model = True  # init from pretrained model
    num_negatives = 5
    batched = False
    model_save_path = None

    ### NQ ###
    if 'nq_10' in dataset_name or dataset_name == 'dureader':
        from ConRAG.dataset.load_nq import load_nq_dataset
        instruction_dataset_train, instruction_dataset_test = load_nq_dataset(
            input_path, 
            4096,
            tokenizer,
            retrieval_aware=False,
            add_noise=False, 
            use_emb='emb' in str(input_path),
        )
    if 'nq_20' in dataset_name or dataset_name == 'dureader':
        from ConRAG.dataset.load_nq import load_nq_dataset
        instruction_dataset_train, instruction_dataset_test = load_nq_dataset(
            input_path, 
            4096,
            tokenizer,
            retrieval_aware=False,
            add_noise=False,
            num_k=20, 
            use_emb='emb' in str(input_path),
        )
    ### HotpotQA ### ### 2Wiki ###
    elif dataset_name == 'hotpotqa' or dataset_name == '2wiki':
        from ConRAG.dataset.load_hotpotqa import load_hotpotqa_dataset
        instruction_dataset_train, instruction_dataset_test = load_hotpotqa_dataset(
            input_path, 
            4096,
            tokenizer,
            retrieval_aware=False,
            add_noise=False, 
            use_cot=False,
            use_emb='emb' in str(input_path),
        )
    ### MuSiQue ###
    elif dataset_name == 'musique':
        from ConRAG.dataset.load_musique import load_musique_dataset
        instruction_dataset_train, instruction_dataset_test = load_musique_dataset(
            input_path, 
            4096,
            tokenizer,
            num_k=20,
            retrieval_aware=False,
            add_noise=False, 
            use_cot=False,
            use_emb='emb' in str(input_path),
        )
    ### MRAG ###
    elif dataset_name == 'mrag':
        from ConRAG.dataset.load_mrag import load_mrag_dataset
        instruction_dataset_train, instruction_dataset_test = load_mrag_dataset(
            input_path, 
            32000,
            tokenizer,
            retrieval_aware=False,
            add_noise=False, 
            use_cot=False,
            use_emb='emb' in str(input_path),
        )
    print('len(train_samples), len(test_samples):', len(instruction_dataset_train), len(instruction_dataset_test))

    ##### Load Retriever
    train_batch_size = int(train_batch_size)
    num_epochs = int(epochs)
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode=pooling)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    model_save_path = 'retrievers/contriever'+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S") if model_save_path is None else model_save_path
    print('model_save_path', model_save_path)

    ##### Init
    train_dataset = RALMDataset(dataset_name, instruction_dataset_train[:], model, num_negatives)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size, drop_last=True)
    train_loss = MultipleNegativesRankingLoss(model=model, similarity_fct=util.dot_score, scale=1)
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
    evaluation_steps = math.ceil(len(train_dataloader) * 1.0) #Evaluate every 50% of the data#40

    ##### Train
    model.fit(train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        # evaluation_steps=evaluation_steps,
        warmup_steps=warmup_steps,
        output_path=None,
        optimizer_params={'lr': lr},
        use_amp=True,        #Set to True, if your GPU supports FP16 cores
    )
    print('dataset_save_path', dataset_save_path)
    new_dataset = get_extractive_results(dataset_name, instruction_dataset_test[:], model, 5)

    pickle.dump(new_dataset, open(dataset_save_path, 'wb'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model with given parameters")
    
    parser.add_argument('--dataset_name', type=str, default='', choices=['nq_10', 'nq_20', 'nq_30', 'hotpotqa', 'musique', '2wiki', 'dureader', 'mrag', 'bioasq', 'lkqa'], help='Name of the dataset')
    parser.add_argument('--input_path', type=str, default=None, help='Path for nq or dureader datasets')
    parser.add_argument('--train_data_path', type=str, default=None, help='Path for the hotpotqa, musique, 2wiki')
    parser.add_argument('--test_data_path', type=str, default=None, help='Path for the hotpotqa, musique, 2wiki')
    parser.add_argument('--dataset_save_path', type=str, default=None, help='Path')
    parser.add_argument('--num_k', type=int, default=5, help='Number of retrieved documents')

    args = parser.parse_args()
    main(**vars(args))