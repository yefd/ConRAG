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

from runner import *

ParaphraseInstructions = [
    'Background: {xrag_token} means the same as',
    "Background: {xrag_token} Can you put the above sentences in your own terms?",
    "Background: {xrag_token} Please provide a reinterpretation of the preceding background text.",
    "These two expressions are equivalent in essence:\n(1) {xrag_token}\n(2)",
    "Background: {xrag_token} is a paraphrase of what?",
    "Background: {xrag_token} Could you give me a different version of the background sentences above?",
    "In other words, background: {xrag_token} is just another way of saying:",
    "You're getting across the same point whether you say background: {xrag_token} or",
    "Background: {xrag_token} After uppacking the ideas in the background information above, we got:",
    "Background: {xrag_token} Please offer a restatement of the background sentences I've just read.",
    "Background: {xrag_token}, which also means:",
    "Strip away the mystery, and you'll find background: {xrag_token} is simply another rendition of:",
    "The essence of background: {xrag_token} is captured again in the following statement:",
]
XRAG_TOKEN = "<xRAG>" 
import re
def extract_question_and_documents(text):
    question_match = re.search(r"^Question:\s*(.*)", text, re.MULTILINE)
    question = question_match.group(1).strip() if question_match else None
    documents = re.findall(r"^\[\d+\](.*?)(?=\[\d+\]|$)", text, re.MULTILINE | re.DOTALL)
    documents = [doc.split('\n\nOnly give me the answer and do not output any other words.')[0].strip() for doc in documents]
    return question, documents
# extract_question_and_documents(instruction_dataset_test[0]['instruction'])
def format_instruction(example):
    if ConRAGRunner.instruction_type == 'instruction':
        text = "### Instruction: \n{instruction}\n\n### Response:\n{output}".format(
            instruction=example['instruction'],
            output=example['output']
        )
    elif ConRAGRunner.instruction_type == 'chat':
        text = "[INST] {instruction} [/INST]{output}\n".format(
            instruction=example['instruction'],
            output=example['output']
        )
    elif ConRAGRunner.instruction_type == 'qwen':
        text = "<|im_start|>user\n{instruction}<|im_end|>{output}\n".format(
            instruction=example['instruction'],
            output=example['output']
        )
    return text
import pathlib
from copy import deepcopy
PROMPTS_ROOT = pathlib.Path('ConRAG/prompts').resolve()

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

def prepare_dataset_stage_1(dataset):
    dataset_new = []
    for data in dataset:
        for i in range(len(data['documents'])):
            sample = {}
            sample['instruction'] = random.choice(ParaphraseInstructions).format_map(dict(xrag_token=XRAG_TOKEN))
            sample['output'] = data['documents'][i]
            sample['embeds'] = data['embeds'][i]
            sample['label'] = data['label'][i]
            dataset_new.append(sample)
    return dataset_new

def prepare_dataset_stage_2(dataset):
    prompt_filename = "qa.prompt"
    with open(PROMPTS_ROOT / prompt_filename) as f:
        prompt_template = f.read().rstrip("\n")
    dataset_new = []
    for data in dataset:
        sample = deepcopy(data)
        sample['instruction_origin'] = format_instruction(sample)
        q_key = 'question' if 'question' in sample.keys() else 'query'
        sample['instruction'] = prompt_template.format(question=sample[q_key], search_results="".join(['<unk>' for i in range(len(sample['documents']))]))
        dataset_new.append(sample)
    return dataset_new

# dataset_new = prepare_dataset(runner.instruction_dataset_test)
# dataset_new = prepare_dataset_stage_2(dataset_new)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import dataclasses
import inspect
import warnings
from functools import wraps
from typing import Callable, Dict, List, Optional, Tuple, Union
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollator,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_utils import unwrap_model
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction


class XRAGTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _prepare_non_packed_dataloader(
        self,
        tokenizer,
        dataset,
        dataset_text_field,
        max_seq_length,
        formatting_func=None,
        add_special_tokens=True,
        remove_unused_columns=True,
    ):
        use_formatting_func = formatting_func is not None and dataset_text_field is None
        print('use_formatting_func', use_formatting_func)
        self._dataset_sanity_checked = False

        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field] if not use_formatting_func else formatting_func(element, False),
                add_special_tokens=add_special_tokens,
                truncation=True,
                padding=False,
                max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,
            )
            outputs_origin = tokenizer(
                # element[dataset_text_field] if not use_formatting_func else formatting_func(element, True),
                element['instruction_origin'],
                # element['instruction_origin'] if not use_formatting_func else XRAGRunner.format_instruction_origin(element),
                add_special_tokens=add_special_tokens,
                truncation=True,
                padding=False,
                max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,
            )

            if use_formatting_func and not self._dataset_sanity_checked:
                if not isinstance(formatting_func(element), list):
                    raise ValueError(
                        "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                    )
                else:
                    self._dataset_sanity_checked = True

            return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"], "embeds": element["embeds"], "label": outputs_origin["input_ids"]}
        signature_columns = ["input_ids", "labels", "attention_mask", "embeds", "answers", "label", "input_ids_origin", "attention_mask"]

        extra_columns = list(set(dataset.column_names) - set(signature_columns))
        print('Remove extra_columns:', extra_columns)

        if not remove_unused_columns and len(extra_columns) > 0:
            warnings.warn(
                "You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with the default collator and yield to errors. If you want to "
                f"inspect dataset other columns (in this case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you used the default collator and create your own data collator in order to inspect the unused dataset columns."
            )
        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names if remove_unused_columns else None,
            num_proc=self.dataset_num_proc,
            batch_size=self.dataset_batch_size,
        )
        print(tokenized_dataset[0].keys(), len(tokenized_dataset[0]['label']), len(tokenized_dataset[0]['input_ids']))
        return tokenized_dataset
from runner import ConRAGRunner, ConRAGTrainer
from ConRAG.models.modeling_xrag import XRAGConfig, XRAGForCausalLM
from datasets import Dataset
from transformers import TrainingArguments

class XRAGRunner(ConRAGRunner):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.use_emb = True

    @staticmethod
    def format_instruction(example, use_origin=False):
        output_texts = []
        dataset_text_field = 'instruction_origin' if use_origin else 'instruction'
        for i in range(len(example[dataset_text_field])):
            if ConRAGRunner.instruction_type == 'instruction':
                text = "### Instruction: \n{instruction}\n\n### Response:\n{output}".format(
                    instruction=example['instruction'][i],
                    output=example['output'][i]
                )
            elif ConRAGRunner.instruction_type == 'chat':
                text = "[INST] {instruction} [/INST]{output}\n".format(
                    instruction=example['instruction'][i],
                    output=example['output'][i]
                )
            elif ConRAGRunner.instruction_type == 'qwen':
                text = "<|im_start|>user\n{instruction}<|im_end|>{output}\n".format(
                    instruction=example['instruction'][i],
                    output=example['output'][i]
                )
            text = text.replace(ConRAGRunner.SEMANTIC_TOKEN, ConRAGRunner.UNK_TOKEN)
            output_texts.append(text)
        return output_texts

    @staticmethod
    def format_instruction_origin(example):
        output_texts = []
        dataset_text_field = 'instruction_origin'
        for i in range(len(example[dataset_text_field])):
            if ConRAGRunner.instruction_type == 'instruction':
                text = "### Instruction: \n{instruction}\n\n### Response:\n{output}".format(
                    instruction=example['instruction'][i],
                    output=example['output'][i]
                )
            elif ConRAGRunner.instruction_type == 'chat':
                text = "[INST] {instruction} [/INST]{output}\n".format(
                    instruction=example['instruction'][i],
                    output=example['output'][i]
                )
            elif ConRAGRunner.instruction_type == 'qwen':
                text = "<|im_start|>user\n{instruction}<|im_end|>{output}\n".format(
                    instruction=example['instruction'][i],
                    output=example['output'][i]
                )
            text = text.replace(ConRAGRunner.SEMANTIC_TOKEN, ConRAGRunner.UNK_TOKEN)
            output_texts.append(text)
        return output_texts

    def load_model(self):
        print('##############################  load_model  ##############################')
        if self.load_from_pretrained:
            raise NotImplementedError
        else:
            config = XRAGConfig(
                model_name_or_path=self.model_name,
                input_dim=self.input_dim,
                hidden_size=self.hidden_size,
                xrag_token=self.UNK_TOKEN,
                xrag_token_id=self.UNK_TOKEN_ID,
                freeze_llm=self.freeze_llm,
                use_flash_att=self.use_flash_att,
                )
            self.model = XRAGForCausalLM(config)
        print(config)
        print(self.model)
    
    def start_training_stage_1(self):
        print('##############################  start_training  ##############################')
        args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            logging_steps=10,
            save_strategy="no",
            learning_rate=6.0e-3,  # cp from xrag
            bf16=True,
            tf32=True,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
            # disable_tqdm=True # disable tqdm since with packing values are in correct
        )
        print(args)
        dataset_train = prepare_dataset(self.instruction_dataset_train[:], self.num_k)
        dataset_train = prepare_dataset_stage_1(dataset_train)
        print(dataset_train[0])
        dataset_train = Dataset.from_list(dataset_train)
        max_seq_length = self.max_prompt_length
        data_collator = None
        trainer = ConRAGTrainer(
            model=self.model,
            train_dataset=dataset_train,
            max_seq_length=max_seq_length,
            tokenizer=self.tokenizer,
            packing=False,
            formatting_func=ConRAGRunner.format_instruction,
            data_collator= data_collator,
            args=args,
        )
        seed_it(42)
        trainer.train()
    
    def start_training_stage_2(self):
        print('##############################  start_training  ##############################')
        args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            logging_steps=10,
            save_strategy="no",
            learning_rate=2.0e-5, ## cp from xrag
            bf16=True,
            tf32=True,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
            # disable_tqdm=True # disable tqdm since with packing values are in correct
        )
        print(args)
        dataset_train = prepare_dataset(self.instruction_dataset_train[:], self.num_k)
        dataset_train = prepare_dataset_stage_2(dataset_train)
        # print(dataset_train[0])
        dataset_train = Dataset.from_list(dataset_train)
        max_seq_length = self.max_prompt_length
        response_tag = 2
        data_collator = DataCollatorForCompletionOnlyLM(self.tokenizer.encode("\nAnswer:", add_special_tokens=False)[response_tag:], tokenizer=self.tokenizer)
        trainer = XRAGTrainer(
            model=self.model,
            train_dataset=dataset_train,
            max_seq_length=max_seq_length,
            tokenizer=self.tokenizer,
            packing=False,
            formatting_func=XRAGRunner.format_instruction,
            data_collator= data_collator,
            args=args,
        )
        seed_it(42)
        trainer.train()

    def get_response(self, sample, prompt_key='instruction'):
        prompt = ConRAGRunner.format_instruction_for_response(sample[prompt_key])
        input_tokens = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=self.max_prompt_length,
                    add_special_tokens=False,
                ).to('cuda')
        embeds = torch.tensor(sample['embeds']).to(input_tokens.input_ids.device)
        if embeds.dim() == 1:
            embeds = embeds.unsqueeze(0)
        inputs = {"input_ids": input_tokens['input_ids'], 'attention_mask': input_tokens['attention_mask'], 'embeds': embeds}

        outputs = self.model.generate(
            inputs=inputs,
            max_new_tokens=100,
            do_sample=False,
            num_beams=self.beam_num if self.use_beam else 1,
            repetition_penalty=1.0,
            length_penalty=1,
            temperature=1.0,
        )
        output_text = self.tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )
        output_text = [text.strip() for text in output_text]
        return output_text

    def run(self):
        self.load_dataset()
        self.load_model()
        self.start_training_stage_1()
        self.start_training_stage_2()
        dataset_test = prepare_dataset(self.instruction_dataset_test[:], self.num_k)
        self.instruction_dataset_test = prepare_dataset_stage_2(dataset_test)
        self.eval()


def main(dataset_name, input_path, train_data_path, test_data_path, dataset_names, paths, **args):
    if dataset_name in ['hotpotqa', 'musique', '2wiki', 'bioasq']:
        input_path = {'train_data_path': train_data_path, 'test_data_path': test_data_path}
    print(args)
    runner = XRAGRunner(
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

    parser.add_argument('--model_name', type=str, required=True, help='Name of LLM')
    parser.add_argument('--load_in_8bit', action='store_true', help='Load in 8-bit precision')
    parser.add_argument('--use_flash_att', action='store_true', help='Load in use_flash_att')
    parser.add_argument('--save_model', action='store_true', help='If set, the trained model will be saved to outputdir')
    parser.add_argument('--output_dir', type=str, required=False, help='Directory to save model')

    parser.add_argument('--model_type', type=str, default='rag', choices=['xrag'], help='model type')
    parser.add_argument('--input_dim', type=int, default=3, help='Input features')
    parser.add_argument('--hidden_size', type=int, default=4096, help='Size of the LLM hidden layer')
    parser.add_argument('--SEMANTIC_TOKEN', type=str, default='<S>', help='Token for retrieval information')
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

