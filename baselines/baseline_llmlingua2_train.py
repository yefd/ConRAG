import torch
import numpy as np
import random, os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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



# nohup python train_llmlingua2.py >  logs/llmlingua/output_nq10.log 2>&1 &
# nohup python train_llmlingua2.py >  logs/llmlingua/output_hotpotqa.log 2>&1 &
# nohup python train_llmlingua2.py >  logs/llmlingua/output_2wiki.log 2>&1 &

# nohup python train_llmlingua2.py >  logs/llmlingua/output_nq10_llama2.log 2>&1 &
# nohup python train_llmlingua2.py >  logs/llmlingua/output_hotpotqa_llama2.log 2>&1 &
# nohup python train_llmlingua2.py >  logs/llmlingua/output_2wiki_llama2.log 2>&1 &

# nohup python train_llmlingua2.py >  logs/llmlingua/output_musique_llama2.log 2>&1 &
# nohup python train_llmlingua2.py >  logs/llmlingua/output_musique_mistral.log 2>&1 &
# nohup python train_llmlingua2.py >  logs/llmlingua/output_nq20_llama2.log 2>&1 &
# nohup python train_llmlingua2.py >  logs/llmlingua/output_nq20_mistral.log 2>&1 &

# nohup python train_llmlingua2.py >  logs/llmlingua/output_mrag_mistral.log 2>&1 &
# nohup python train_llmlingua2.py >  logs/llmlingua/output_mrag_llama2.log 2>&1 &

# at least two keys: *idx* and *prompt*. 
# dataset_name = 'hotpotqa'
# dataset_name = '2wiki'
# dataset_name = 'musique'
# dataset_name = 'nq10'
# dataset_name = 'nq20'
dataset_name = 'mrag'
llama2 = ''
# llama2 = '_llama2'

import argparse
from collections import defaultdict

import numpy as np
import torch



data_path = f'output/longllmlingua/{dataset_name}_label_word{llama2}.pt'
# save_path = f'output/longllmlingua/{dataset_name}_train_result{llama2}.pt'


model_name = "models/xlm-roberta-large"


import argparse
import os, sys
sys.path.append('baselines/LLMLingua/experiments/llmlingua2/model_training/')
import random
import time

import torch
from sklearn.metrics import accuracy_score
from torch import cuda
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer
from utils import TokenClfDataset

MAX_LEN = 512
MAX_GRAD_NORM = 10
label_type = "word_label"
num_epoch = 10
lr = 1e-5
batch_size = 10
# os.makedirs(os.path.dirname(save_path), exist_ok=True)



def train(epoch):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    model.train()

    for idx, batch in enumerate(train_dataloader):
        t = time.time()
        ids = batch["ids"].to(device, dtype=torch.long)
        mask = batch["mask"].to(device, dtype=torch.long)
        targets = batch["targets"].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
        loss, tr_logits = outputs.loss, outputs.logits
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        flattened_targets = targets.view(-1)
        active_logits = tr_logits.view(-1, model.num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1)
        active_accuracy = mask.view(-1) == 1
        targets = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        tr_preds.extend(predictions)
        tr_labels.extend(targets)

        tmp_tr_accuracy = accuracy_score(
            targets.cpu().numpy(), predictions.cpu().numpy()
        )
        tr_accuracy += tmp_tr_accuracy

        if idx % 100 == 0:
            loss_step = tr_loss / nb_tr_steps
            acc_step = tr_accuracy / nb_tr_steps

        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=MAX_GRAD_NORM
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    tr_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {tr_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")


def test(model, eval_dataloader):
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []

    with torch.no_grad():
        for idx, batch in enumerate(eval_dataloader):
            ids = batch["ids"].to(device, dtype=torch.long)
            mask = batch["mask"].to(device, dtype=torch.long)
            targets = batch["targets"].to(device, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
            loss, eval_logits = outputs.loss, outputs.logits

            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += targets.size(0)

            flattened_targets = targets.view(-1)
            active_logits = eval_logits.view(-1, model.num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1)
            active_accuracy = mask.view(-1) == 1
            targets = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            eval_labels.extend(targets)
            eval_preds.extend(predictions)

            tmp_eval_accuracy = accuracy_score(
                targets.cpu().numpy(), predictions.cpu().numpy()
            )
            eval_accuracy += tmp_eval_accuracy

    labels = [label.item() for label in eval_labels]
    predictions = [pred.item() for pred in eval_preds]

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")

    return eval_accuracy



tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=2, ignore_mismatched_sizes=True
)
model.to('cuda')


device = "cuda" if cuda.is_available() else "cpu"
data = torch.load(data_path)
assert len(data["origin"]) == len(data["labels"])
text_label = [(text, label) for text, label in zip(data["origin"], data["labels"])]
random.shuffle(text_label)
train_data = text_label[: int(len(text_label) * 0.8)]
val_data = text_label[int(len(text_label) * 0.8) :]
train_text = [text for text, label in train_data]
train_label = [label for text, label in train_data]
val_text = [text for text, label in val_data]
val_label = [label for text, label in val_data]


train_dataset = TokenClfDataset(
    train_text, train_label, MAX_LEN, tokenizer=tokenizer, model_name=model_name
)
val_dataset = TokenClfDataset(
    val_text, val_label, MAX_LEN, tokenizer=tokenizer, model_name=model_name
)

print(f"len taining set: {len(train_dataset)}, len validation set: {len(val_dataset)}")
print(train_dataset[0])
for token, label in zip(
    tokenizer.convert_ids_to_tokens(train_dataset[0]["ids"][:30]),
    train_dataset[0]["targets"][:30],
):
    print("{0:10}  {1}".format(token, label.item()))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


ids = train_dataset[0]["ids"].unsqueeze(0)
mask = train_dataset[0]["mask"].unsqueeze(0)
targets = train_dataset[0]["targets"].unsqueeze(0)
ids = ids.to(device)
mask = mask.to(device)
targets = targets.to(device)
outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
initial_loss = outputs[0]
print(initial_loss)

tr_logits = outputs[1]
print(tr_logits.shape)
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)



best_acc = 0
num_epoch = 1
for epoch in tqdm(range(num_epoch)):
    print(f"Training epoch: {epoch + 1}")
    train(epoch)
    acc = test(model, val_dataloader)
    if acc > best_acc:
        best_acc = acc
        # torch.save(model.state_dict(), args.save_path)
save_path = f'output/longllmlingua/roberta_{dataset_name}{llama2}/'
print('save model', save_path)
os.makedirs(os.path.dirname(save_path), exist_ok=True)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)


# torch.save(model.state_dict(), save_path)


import argparse
import copy
import json
import os, sys
sys.path.append('baselines/LLMLingua/')
import time

from tqdm import tqdm

from llmlingua.prompt_compressor import PromptCompressor

from transformers import AutoTokenizer
model_name = 'models/Mistral-7B-Instruct-v0.3' if llama2 == '' else 'models/Llama-2-7b-chat-hf'
# model_name = 'models/longchat-7b-v1.5-32k'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token


import sys
import os
# sys.path.append(os.path.dirname(os.path.abspath(os.getcwd())))

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

if dataset_name == 'mrag':
    from ConRAG.dataset.load_mrag import *
    input_path = 'noise/ConRAG/dataset/multihop_rag/multihop_rag_emb.pkl'
    instruction_dataset_train, instruction_dataset_test = load_mrag_dataset(
        input_path, 
        retrieval_aware=True,
        use_cot=False,
        use_emb=False,
        standard_prompt=True,
        num_k=10,
        max_prompt_length=32000, tokenizer=tokenizer)
    data = [{
        'idx': str(i),
        'prompt': instruction_dataset_test[i]['instruction'],
        'question': instruction_dataset_test[i]['query'],
        'answers': [instruction_dataset_test[i]['answer']],
    } for i in range(len(instruction_dataset_test))]
    print(f"num data: {len(data)}")

load_origin_from = ""
compression_rate = 0.5
force_tokens = "\n,?,!,."
save_path = f'output/longllmlingua/{dataset_name}_compress_result{llama2}.json'
load_key = "prompt"
save_key = "compressed_prompt"
model_name = f'output/longllmlingua/roberta_{dataset_name}{llama2}'


os.makedirs(os.path.dirname(save_path), exist_ok=True)
if force_tokens is not None:
    force_tokens = [
        str(item).replace("\\n", "\n") for item in force_tokens.split(",")
    ]
else:
    force_tokens = []
print(f"force tokens: {force_tokens}")

compressor = PromptCompressor(
    model_name=model_name,
    model_config={},
    use_llmlingua2=True,
)
results = {}
results_list = []
total_time = 0

if os.path.exists(save_path):
    results = json.load(open(save_path))

print('len results', len(results))
print('len(data)', len(data))

for sample in tqdm(data[:]):
    idx = str(sample["idx"])
    prompt = copy.deepcopy(sample[load_key])
    if prompt is None:
        continue
    c = prompt.split("\n\n")
    instruction, question = c[0], c[-1]
    demonstration = "\n".join(c[1:-1])
    if idx in results or str(idx) in results:
        print(f"{idx}-th sample is processed")
        continue
    t = time.time()
    # comp_dict = compressor.compress_prompt(
    #     demonstration.split("\n"),
    #     instruction=instruction,
    #     question=question,
    #     target_token=500,
    #     condition_compare=True,
    #     condition_in_question="after",
    #     rank_method="longllmlingua",
    #     use_sentence_level_filter=False,
    #     context_budget="+100",
    #     dynamic_context_compression_ratio=0.4,  # enable dynamic_context_compression_ratio
    #     reorder_context="sort",
    #     use_token_level_filter=False
    # )
    comp_dict = compressor.compress_prompt(
        prompt,
        rate=compression_rate,
        use_token_level_filter=True,
        use_context_level_filter=False,
        target_context=-1,
        context_level_rate=1.0,
        context_level_target_token=-1,
        force_tokens=force_tokens,
        drop_consecutive=True,
        force_reserve_digit=False,
    )
    total_time += time.time() - t
    comp = comp_dict["compressed_prompt"]
    # print(comp)
    # print(prompt)
    comp_list = comp_dict["compressed_prompt_list"]

    new_sample = copy.deepcopy(sample)
    new_sample[save_key] = comp
    if comp_list is not None and load_key == "prompt_list":
        new_sample["compressed_prompt_list"] = comp_list
        # print(len(new_sample["prompt_list"]), len(new_sample["compressed_prompt_list"]))

    results[idx] = new_sample
    # json.dump(
    #     results,
    #     open(save_path, "w", encoding="utf8"),
    #     indent=4,
    #     ensure_ascii=False,
    # )

print(save_path, total_time)
json.dump(
    results, open(save_path, "w", encoding="utf8"), indent=4, ensure_ascii=False
)
