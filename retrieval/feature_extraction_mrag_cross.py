import torch
import numpy as np
import random, os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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


from datasets import load_dataset

dataset = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG")
corpus = load_dataset("yixuantt/MultiHopRAG", "corpus")


import pickle
from xopen import xopen
from tqdm import tqdm
import json
import argparse
from copy import deepcopy
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers.evaluation import RerankingEvaluator
from tqdm import tqdm, trange

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = r'bge-reranker-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).cuda()
model.eval()

pairs = [['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to('cuda')
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    print(scores)

def get_rank_features(query_text, corpus_text, top_k, retrieval_model):
    scores = []
    hidden_states = []
    for d in corpus_text:
        pair = [[query_text, d]]
        inputs = tokenizer(pair, padding=True, truncation=True, return_tensors='pt', max_length=512).to('cuda')
        output = model(**inputs, output_hidden_states=True, return_dict=True)
        score = output.logits.view(-1, ).detach().float().cpu().numpy() # [1]
        scores.append(score)
        last_hidden_state = output.hidden_states[-1].mean(dim=1).detach() # [1, 1024]
        hidden_states.append(last_hidden_state.cpu())
    scores = np.concatenate(scores, axis=0)
    hidden_states = torch.cat(hidden_states, dim=0)
    # print(scores)
    rank_list = scores.argsort().tolist()[::-1][:top_k]
    rank_score = [scores[idx] for idx in rank_list]
    
    precedent_sim = [1]
    for i in range(1, len(rank_list)):
        precedent_score = np.array([scores[idx] for idx in rank_list[:i]])
        precedent_weight = np.exp(precedent_score)/np.exp(precedent_score).sum()
        w = torch.Tensor(precedent_weight, device='cpu').reshape(1, precedent_score.shape[0])
        precedent_embs = [hidden_states[idx].reshape(1, hidden_states.shape[1]).cpu() for idx in rank_list[:i]]
        precedent_emb = torch.mm(w, torch.cat(precedent_embs))
        cur_emb = hidden_states[rank_list[i]].cpu()
        score = cos_sim(cur_emb, precedent_emb)[0][0]
        precedent_sim.append(score)
    
    rank_emb = [hidden_states[i] for i in rank_list]
    nb_sim = [cos_sim(rank_emb[0], rank_emb[1]).cpu()]
    for i in range(1, len(rank_list)-1):
        nb_sim.append((cos_sim(rank_emb[i], rank_emb[i-1]).cpu() + cos_sim(rank_emb[i], rank_emb[i+1]).cpu()) / 2)
    nb_sim.append(cos_sim(rank_emb[-2], rank_emb[-1]).cpu())
    nb_sim = [s.squeeze() for s in nb_sim]
    # print(cosine_score, rank_list, rank_score)
    return np.array(rank_score), rank_list, np.array(precedent_sim), np.array(nb_sim), rank_emb

from tqdm import tqdm
def _init_dataset_mrag(dataset, corpus, idx, retrieval_model, top_k):
    dataset_new = []
    corpus_text = ['Title: '+ctx['title'] +'\n' + ctx['body'] for ctx in corpus]
    for i in tqdm(idx):
        query_text = dataset[i]['query']
        evidence_list = dataset[i]['evidence_list']
        evidence_titles = [e['title'] for e in evidence_list]
        rank_score, rank_list, precedent_sim, nb_sim, rank_emb = get_rank_features(query_text, corpus_text, top_k, retrieval_model)
        ctxs = []
        for r in range(len(rank_list)):
            j = rank_list[r]
            ctx_text = corpus_text[j]
            ctx = {'id': j, 'text': ctx_text, 'label': 1 if corpus[j]['title'] in evidence_titles else 0}
            ctx['emb'] = rank_emb[r].tolist()
            ctx['rerank_score'] = float(rank_score[r])
            ctx['rerank_nb_score'] = float(nb_sim[r])
            ctx['rerank_precedent_score'] = float(precedent_sim[r])
            ctxs.append(ctx)
        data = deepcopy(dataset[i])
        data['ctxs'] = ctxs
        dataset_new.append(data)
    return dataset_new

def init_dataset_mrag(dataset, corpus, retrieval_model, top_k=10, dataset_seed=42):
    dataset = dataset['train']
    corpus = corpus['train']
    seed_it(dataset_seed)
    all_index = list(range(len(dataset)))
    train_index = np.sort(random.sample(all_index, int(len(all_index)*0.8))).tolist()
    test_index = np.sort(list(set(all_index).difference(set(train_index)))).tolist()[:]
    print(f'prepare dataset, train size: {len(train_index)}, test size: {len(test_index)}')

    dataset_new = _init_dataset_mrag(dataset, corpus, all_index[:], retrieval_model, top_k)
    return dataset_new

mrag_dataset = init_dataset_mrag(dataset, corpus, model, top_k=10)

save_path = 'dataset/multihop_rag/multihop_rag_bge_rerank_emb.pkl'
with open(save_path, 'wb') as fin:
    pickle.dump(mrag_dataset, fin)
    fin.close()
print('save_path', save_path)




