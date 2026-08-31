import torch
import numpy as np
import random, os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

from runner import ConRAGRunner, ConRAGTrainer
from datasets import Dataset
from transformers import TrainingArguments
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from torch.utils.data import DataLoader
from transformers import default_data_collator
from ConRAG.utils.metrics import *
from runner import *
import pathlib
import gc
from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    DataCollatorForCompletionOnlyLM,
    create_reference_model,
)
from torch.utils.data import DataLoader
from transformers import default_data_collator

PROMPTS_ROOT = pathlib.Path('ConRAG/prompts').resolve()

def custom_collate_fn(features):
    batch = default_data_collator([f for f in features if isinstance(f, dict)])
    for field in ["answer", "documents"]:
        if field in features[0]:
            batch[field] = [f[field] for f in features]
    
    return batch

def prepare_non_packed_dataloader(
    tokenizer,
    dataset,
    dataset_text_field,
    max_seq_length,
    formatting_func=None,
    add_special_tokens=True,
    remove_unused_columns=True,
    shuffle=False,
    batch_size=1,
):
    def tokenize(element):
        outputs = tokenizer(
            formatting_func(element),
            # element[dataset_text_field],
            add_special_tokens=add_special_tokens,
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            return_overflowing_tokens=False,
            return_length=False,
        )
        # print(element['answers'])
        answers_outputs = tokenizer(
            [e[0] for e in element['answers']] if 'answers' in element.keys() else [e for e in element['answer']],
            add_special_tokens=add_special_tokens,
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            return_overflowing_tokens=False,
            return_length=False,
        )
        tokenized_element = {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"], "label": element["label"], "answers_ids": answers_outputs["input_ids"]}
        # for field in ["question", "answer", "documents", "answers"]:
        #     if field in element:
        #         tokenized_element[field] = element[field]
        return tokenized_element
    signature_columns = ["input_ids", "labels", "attention_mask", "label", "answers_ids"]

    extra_columns = list(set(dataset.column_names) - set(signature_columns))
    print('remove extra_columns:', extra_columns)
    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
        batch_size=batch_size,
    )

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate_fn,
    )
    return dataloader, tokenized_dataset
# dataloader, dataset = prepare_non_packed_dataloader(tokenizer, dataset_train, 'instruction', max_prompt_length, format_instruction_for_ppo)
from copy import deepcopy

def prepare_dataset(dataset, num_k=10):
    dataset_new = []
    for data in dataset:
        sample = deepcopy(data)
        question, documents= extract_question_and_documents(sample['instruction'])
        if len(documents) != num_k:
            raise ValueError
        sample['documents'] = documents
        dataset_new.append(sample)
    return dataset_new

import re
def extract_question_and_documents(text):
    question_match = re.search(r"^Question:\s*(.*)", text, re.MULTILINE)
    question = question_match.group(1).strip() if question_match else None
    documents = re.findall(r"^\[\d+\](.*?)(?=\[\d+\]|$)", text, re.MULTILINE | re.DOTALL)
    documents = [doc.split('\n\nOnly give me the answer and do not output any other words.')[0].strip() for doc in documents]
    return question, documents

def format_instruction_for_sft(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        output = ''.join([f'<|reserved_special_token_{i+1}|>' for i, value in enumerate(example['label'][i]) if value == 1]) + '<|end_of_text|>'
        text = tokenizer.apply_chat_template([{"role": "user", "content": example['instruction'][i]}, {"role": "assistant", "content": output}], tokenize=False)
        output_texts.append(text)
    return output_texts

def format_instruction_for_ppo(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = tokenizer.apply_chat_template([{"role": "user", "content": example['instruction'][i]}], tokenize=False) + '<|start_header_id|>assistant<|end_header_id|>\n\n'
        output_texts.append(text)
    return output_texts

def get_qa_instruction(question, documents, tokenizer, idx):
    prompt_filename = "qa.prompt"
    with open(PROMPTS_ROOT / prompt_filename) as f:
        prompt_template = f.read().rstrip("\n")
    formatted_documents = []
    for document_index in idx:
        document = documents[document_index]
        document_prompt = f"[{document_index+1}]{document}"
        formatted_documents.append(document_prompt)
    return prompt_template.format(question=question, search_results="\n".join(formatted_documents))
# t
def calculate_metrics_average(data):
    metrics_sum = {}
    count = len(data)
    for item in data:
        metrics = item.get('metrics', {})
        for key, value in metrics.items():
            if key in metrics_sum:
                metrics_sum[key] += value
            else:
                metrics_sum[key] = value
    metrics_avg = {key: value / count for key, value in metrics_sum.items()}
    print(metrics_avg)

from transformers import LogitsProcessorList, LogitsProcessor, PrefixConstrainedLogitsProcessor

class FixedTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        LogitsProcessor.__init__(self)
        allowed_tokens = ''.join([f'<|reserved_special_token_{i}|>' for i in range(1, 11)]) + '<|end_of_text|>'
        # self.allowed_token_ids = tokenizer.convert_tokens_to_ids(list(allowed_tokens)+[tokenizer.eos_token])
        self.allowed_token_ids = tokenizer.encode(allowed_tokens) + [tokenizer.eos_token_id]
        self.allowed_token_ids = list(set(self.allowed_token_ids))
        print(self.allowed_token_ids)

    def __call__(self, input_ids, scores):
        mask = torch.ones(scores.shape, dtype=torch.bool, device=scores.device)
        mask[:, self.allowed_token_ids] = False
        scores = scores.masked_fill(mask, -float("inf"))
        return scores

class BGMRunner(ConRAGRunner):
    def __init__(
        self,
        bridge_model_name,
        reward_model_name,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.use_emb = True
        self.bridge_model_name = bridge_model_name
        self.reward_model_name = reward_model_name
        self.model_name = bridge_model_name

    def load_model(self):
        print('##############################  load_model  ##############################')
        if self.load_from_pretrained:
            raise NotImplementedError
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.bridge_model_name, trust_remote_code=True, max_length=self.max_prompt_length)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.bridge_model = AutoModelForCausalLM.from_pretrained(self.bridge_model_name, trust_remote_code=True, device_map="cuda", ) # 
            self.reward_tokenizer = AutoTokenizer.from_pretrained(self.reward_model_name, trust_remote_code=True, max_length=self.max_prompt_length)
            self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token
            self.reward_model = AutoModelForCausalLM.from_pretrained(self.reward_model_name, trust_remote_code=True, device_map="cuda", attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)

            logits_processor = LogitsProcessorList()
            logits_processor.append(FixedTokenLogitsProcessor(self.tokenizer))
            self.logits_processor = logits_processor
        print(self.bridge_model)
        print(self.reward_model)

    def get_reward_output(self, bridge_output, question, documents):
        # Tokenize and generate a reference organization from the LLM (e.g., LLaMA2)
        if bridge_output == '<|end_of_text|>':
            return ''
        try:
            bridge_output = bridge_output.replace('<|start_header_id|>assistant<|end_header_id|>', '').split('<|end_of_text|>')[0].strip()
            output_tokens = tokenizer.encode(bridge_output, add_special_tokens=False)
            output_idx = [token2idx[t] for t in output_tokens]
        except Exception as e:
            # print(e)
            return ''
        idx_new = []
        for i in output_idx:
            if i >= len(documents):
                return ''
            if i not in idx_new:
                idx_new.append(i)
        output_idx = idx_new
        instruction = ConRAGRunner.format_instruction_for_response(get_qa_instruction(question, documents, self.reward_tokenizer, output_idx))
        reward_inputs = self.reward_tokenizer(instruction,
                                            return_tensors="pt",
                                            padding=False,
                                            truncation=True,
                                            max_length=self.max_prompt_length,
                                            add_special_tokens=False,
                                            ).to(self.reward_model.device)

        with torch.no_grad():
            embed_tokens = self.reward_model.get_input_embeddings()
            inputs_embeds = embed_tokens(reward_inputs.input_ids)
            ref_ids = self.reward_model.generate(inputs_embeds=inputs_embeds,
                                            attention_mask=reward_inputs.attention_mask,
                                            max_new_tokens=100,
                                            do_sample=False,
                                            num_beams=1,
                                            repetition_penalty=1.0,
                                            length_penalty=1,
                                            temperature=1.0,
                                            pad_token_id=tokenizer.eos_token_id,
                                            )
        ref_output = self.reward_tokenizer.decode(ref_ids[0], skip_special_tokens=True)
        return ref_output  # Return BLEU score as the reward
    # t = get_reward_output('<|reserved_special_token_1|>', instruction_dataset_test[0]['question'], [f"(Title: {d['title']}) {d['text']}" for d in instruction_dataset_test[0]['ctxs']])

    def start_training_stage_sft(self):
        print('##############################  start_training  ##############################')
        args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            logging_steps=10,
            save_strategy="no",
            learning_rate=1e-5,
            bf16=True,
            tf32=True,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
        )
        print(args)
        dataset_train = self.instruction_dataset_train[:]
        dataset_train = Dataset.from_list(dataset_train)
        data_collator = DataCollatorForCompletionOnlyLM(self.tokenizer.encode("\nRelevant documents ids:", add_special_tokens=False)[2:], tokenizer=self.tokenizer)
        print(dataset_train[0])
        dataset_train = Dataset.from_list(dataset_train)
        max_seq_length = self.max_prompt_length
        data_collator = None
        trainer = SFTTrainer(
            model=self.bridge_model,
            train_dataset=dataset_train,
            max_seq_length=max_seq_length,
            tokenizer=self.tokenizer,
            packing=False,
            formatting_func=format_instruction_for_sft,
            data_collator= data_collator,
            args=args,
        )
        seed_it(42)
        trainer.train()

    def start_training_stage_ppo(self):
        print('##############################  start_training  ##############################')
        config = PPOConfig(
            model_name=self.bridge_model_name,
            batch_size=1,
            ppo_epochs=1,
            # steps=200,
            mini_batch_size=1,
            learning_rate=1.41e-5,
            log_with="tensorboard",
        )
        bridge_model = AutoModelForCausalLMWithValueHead.from_pretrained(self.bridge_model)
        reference_model = create_reference_model(bridge_model)
        tokenizer = self.tokenizer
        datset_trian = prepare_dataset(self.instruction_dataset_train[:], self.num_k)
        dataset_train = Dataset.from_list(datset_trian)
        dataloader, dataset = prepare_non_packed_dataloader(tokenizer, dataset_train, 'instruction', self.max_prompt_length, format_instruction_for_ppo, batch_size=config.batch_size)

        logits_processor = self.logits_processor
        ppo_trainer = PPOTrainer(config, bridge_model, reference_model, tokenizer, dataset,)
        self.ppo_trainer = ppo_trainer
        generation_kwargs = {
            "max_new_tokens": 32,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id":tokenizer.eos_token_id,
            "logits_processor": logits_processor,
            "repetition_penalty": 1.0,
            "temperature": 1.0,
        }
        self.eval()
        gc.collect()
        torch.cuda.empty_cache()
        self.bridge_model.train()
        for epoch in range(ppo_trainer.config.ppo_epochs):
            for batch in tqdm(dataloader):
                response_ids = ppo_trainer.generate(
                        query_tensor=batch["input_ids"][0].to('cuda'),
                        **generation_kwargs
                )
                response_texts = tokenizer.batch_decode(response_ids[:, batch['input_ids'].shape[1]:], skip_special_tokens=False)[0]
                if response_texts == '' or response_texts == '<|end_of_text|>':
                    reward = torch.tensor(-1.0, device=response_ids.device)
                else:
                    instruction_texts = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
                    question, documents = extract_question_and_documents(instruction_texts[0])
                    
                    with torch.no_grad():
                        reward_output = self.get_reward_output(response_texts, question, documents)
                    answer = self.tokenizer.batch_decode(batch['answers_ids'], skip_special_tokens=True)
                    # print('answer', answer)
                    if answer == '':
                        reward = torch.tensor(-1.0, device=response_ids.device)
                    else:
                        example = {'answers': answer, 'model_answer': reward_output}
                        metrics, _ = get_metrics_for_example(example, METRICS_EN)
                        reward = torch.Tensor(list(metrics.values())).to(response_ids.device).sum() * 5
                    # print(reward)
                stats = ppo_trainer.step([batch["input_ids"][0]], [response_ids[0]], [reward])
                print(f"Epoch {epoch + 1}, reward: {reward}") # , stats: {stats}

    def eval(self):
        print('##############################  evaluation_from_list  ##############################')
        self.bridge_model.eval()
        tokenizer = self.tokenizer
        logits_processor = self.logits_processor
        max_prompt_length = self.max_prompt_length
        ppo_trainer = self.ppo_trainer
        dataset_test = Dataset.from_list(self.instruction_dataset_test[:])
        dataloader_test, dataset_test = prepare_non_packed_dataloader(tokenizer, dataset_test, 'instruction', max_prompt_length, format_instruction_for_ppo)
        generation_kwargs = {
            "max_new_tokens": 32,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "logits_processor": logits_processor,
            "repetition_penalty": 1.0,
            "temperature": 1.0,
        }
        results = []
        for batch in tqdm(dataloader_test):
            with torch.no_grad():
                response_ids = ppo_trainer.generate(
                        query_tensor=batch["input_ids"][0].to('cuda'),
                        **generation_kwargs
                    )
                response_texts = tokenizer.batch_decode(response_ids[:, batch['input_ids'].shape[1]:], skip_special_tokens=False)[0]
                instruction_texts = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
                question, documents = extract_question_and_documents(instruction_texts[0])
                reward_output = self.get_reward_output(response_texts, question, documents)
                answer = tokenizer.batch_decode(batch['answers_ids'], skip_special_tokens=True)
                example = {'answers': answer, 'model_answer': reward_output}
                metrics, _ = get_metrics_for_example(example, METRICS_EN)
                results.append({'answer': answer, 'reward_output': reward_output, 'metrics': metrics, 'bridge_output': response_texts})
        calculate_metrics_average(results)
        if self.save_results:
            save_pkl_file = f'res_bgm_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if not os.path.exists(f'output/{self.dataset_name}'):
                os.makedirs(f'output/{self.dataset_name}')
            pkl_save_path = f'output/{self.dataset_name}/{save_pkl_file}.pkl'
            print('results_save_path', pkl_save_path)
            with open(pkl_save_path, 'wb') as f:
                pickle.dump(results, f)
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
        res = [r['reward_output'] for r in results]
        # ans = [r['answer'] for r in results]
        m = evaluation_from_list(res[:], gt_ans[:len(res)], self.dataset_name)
        # bridge_output = [r['bridge_output'] for r in results]
        # print(res, ans, gt_ans, bridge_output)

    def run(self):
        self.load_dataset()
        global token2idx
        token2id = self.tokenizer.encode(''.join([f'<|reserved_special_token_{i}|>' for i in range(1, 21)]))[1:]
        token2idx = {value: idx for idx, value in enumerate(token2id)}
        global tokenizer
        tokenizer = self.tokenizer
        self.load_model()
        self.start_training_stage_sft()
        gc.collect()
        torch.cuda.empty_cache()
        self.start_training_stage_ppo()
        gc.collect()
        torch.cuda.empty_cache()
        self.eval()

def main(dataset_name, input_path, train_data_path, test_data_path, dataset_names, paths, **args):
    if dataset_name in ['hotpotqa', 'musique', '2wiki', 'bioasq']:
        input_path = {'train_data_path': train_data_path, 'test_data_path': test_data_path}
    print(args)
    runner = BGMRunner(
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
    parser.add_argument('--standard_prompt', action='store_true', help='Use standard prompt')

    parser.add_argument('--bridge_model_name', type=str, required=False, help='Name of LLM')
    parser.add_argument('--reward_model_name', type=str, required=True, help='Name of LLM')
    parser.add_argument('--load_in_8bit', action='store_true', help='Load in 8-bit precision')
    parser.add_argument('--use_flash_att', action='store_true', help='Load in use_flash_att')
    parser.add_argument('--save_model', action='store_true', help='If set, the trained model will be saved to outputdir')
    parser.add_argument('--output_dir', type=str, required=False, help='Directory to save model')

    parser.add_argument('--model_type', type=str, default='rag', choices=['bgm'], help='model type')
    parser.add_argument('--input_dim', type=int, default=3, help='Input features')
    parser.add_argument('--hidden_size', type=int, default=4096, help='Size of the LLM hidden layer')
    parser.add_argument('--RETRIEVAL_TOKEN', type=str, default='<R>', help='Token for retrieval information')
    parser.add_argument('--UNK_TOKEN', type=str, default='<unk>', help='Token for unknown tokens')
    parser.add_argument('--UNK_TOKEN_ID', type=int, default=0, help='Unknown token ID')

    parser.add_argument('--num_k', type=int, default=10, help='Number of retrieved documents')
    parser.add_argument('--use_training', action='store_true', help='Use for training')
    parser.add_argument('--freeze_llm', action='store_true', help='Freeze LLM')
    parser.add_argument('--load_from_pretrained', action='store_true', help='Load from ConRAG pretrained model')
    parser.add_argument('--pretrained_model_name', type=str, required=False, help='Name of ConRAG pretrained model')
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA')
    parser.add_argument('--use_cot', action='store_true', help='Use CoT')
    parser.add_argument('--num_train_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=1, help='Batch size per device')
    parser.add_argument('--completion_only', action='store_true', help='Use DataCollatorForCompletionOnlyLM')

    parser.add_argument('--use_evaluation', action='store_true', help='Use for evaluation')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Maximum new tokens')
    parser.add_argument('--use_beam', action='store_true', help='Use beam search')
    parser.add_argument('--beam_num', type=int, default=5, help='Number of beams in beam search')
    parser.add_argument('--save_results', action='store_true', help='Save results')

    parser.add_argument('--instruction_type', default='llama', choices=['chat', 'instruction', 'qwen'], help='instruction_type, llama or mistral')

    args = parser.parse_args()
    main(**vars(args))




