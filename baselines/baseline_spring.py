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
from ConRAG.models.modeling_spring import SpringTuningForCausalLM
from datasets import Dataset
from transformers import TrainingArguments
from peft import PromptTuningInit, PromptTuningConfig, TaskType
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel

import linecache
import json
import random
from torch.utils.data import Dataset
from copy import deepcopy

def tokenize_batch_for_finetune(batch, tokenizer, max_length: int = 4096):
    # prompt, completion, reference
    input_output_texts = [sample["reference"] + sample["prompt"] + " " + sample["completion"] + " " + tokenizer.eos_token for sample in batch]
    completion = [sample["completion"] + " " + tokenizer.eos_token for sample in batch]
    spring_insert_text = [sample["prompt"] + " " + sample["completion"] + " " + tokenizer.eos_token for sample in batch]
    data = tokenizer(input_output_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length, add_special_tokens=True)
    data_completion = tokenizer(completion, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length, add_special_tokens=False)
    data_spring_insert_text = tokenizer(spring_insert_text, padding=False, truncation=True, max_length=max_length, add_special_tokens=False)
    len_spring_insert_text = [len(data) for data in data_spring_insert_text["input_ids"]]
    data_mask_reverse = 1 - data_completion["attention_mask"]
    data_mask = data_mask_reverse * -100
    data["labels"] = data["input_ids"].clone()
    data["labels"] *= data_completion["attention_mask"]
    data["labels"] += data_mask
    data["insert_position"] = max_length - torch.tensor(len_spring_insert_text)
    data = {k: v.cuda() for k, v in data.items()}
    return data

def load_tokens(model, tokenizer, new_tokens_weights):
    new_tokens_length = new_tokens_weights.shape[0]
    # expand vocabulary
    new_tokens = [f"[ref{i+1}]" for i in range(new_tokens_length)]
    tokenizer.add_tokens(new_tokens)
    # get original embedding weight matrix
    embedding_layer = model.get_input_embeddings()
    embedding_weights = embedding_layer.weight
    original_vocab_size, embedding_dim = embedding_weights.shape
    
    # create new embedding matrix
    new_vocab_size = original_vocab_size + new_tokens_length
    new_embedding_weights = torch.zeros(new_vocab_size, embedding_dim)

    # copy original embeddings to the new weights
    new_embedding_weights[:original_vocab_size, :] = embedding_weights

    # append virtual token embeddings to the new weights
    for token, embedding in zip(new_tokens, new_tokens_weights):
        token_id = tokenizer.convert_tokens_to_ids(token)
        new_embedding_weights[token_id] = embedding
    
    # update the embedding table
    # note: we should avoid using the function resize_token_embeddings() because this function will also change the lm_head of the model
    embedding_layer.weight.data = new_embedding_weights.to(torch.bfloat16).to('cuda')
    return tokenizer, model

class FineTuningQADataset(Dataset):
    def __init__(self, dataset, ret_passages=10):
        super(FineTuningQADataset, self).__init__()
        self.dataset = dataset
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        data = sample['instruction'].split('\n\n')
        ret_passages = '\n\n'.join(data[:-1])
        question_prompt = data[-1]
        
        if isinstance(sample["output"], list):
            target = random.choice(sample["output"])
        else:
            target = sample["output"]

        batch = {
            "reference": ret_passages,
            "prompt": question_prompt,
            "completion": target
        }

        return batch

    def __len__(self):
        return len(self.dataset)

class SpringRunner(ConRAGRunner):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.use_emb = True

    def load_model(self):
        print('##############################  load_model  ##############################')
        if self.load_from_pretrained:
            raise NotImplementedError
        else:
            peft_config = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                prompt_tuning_init=PromptTuningInit.TEXT,
                num_virtual_tokens=50,
                prompt_tuning_init_text="According to the previous relevant passages, please answer the following question. Only return the answer without any other words.",
                tokenizer_name_or_path=self.model_name,
            )
            model = AutoModelForCausalLM.from_pretrained(self.model_name, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
            self.model = SpringTuningForCausalLM(model, peft_config).to('cuda')
        print(peft_config)
        print(self.model)
    
    def start_training(self):
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
            learning_rate=1.0e-4,  # cp from spring
            bf16=True,
            tf32=True,
            max_grad_norm=0.3,
            lr_scheduler_type="constant",
            dataloader_pin_memory=False,
            do_eval=False,
            save_total_limit=6,
            warmup_ratio=0.02,
            label_names=["completion"],
            remove_unused_columns=False,
            # disable_tqdm=True # disable tqdm since with packing values are in correct
        )
        print(args)
        dataset_train = FineTuningQADataset(self.instruction_dataset_train[:8000])
        print(dataset_train[0])
        max_seq_length = self.max_prompt_length
        data_collator = lambda data: tokenize_batch_for_finetune(data, tokenizer=self.tokenizer, max_length=self.max_prompt_length)
        trainer = ConRAGTrainer(
            model=self.model,
            train_dataset=dataset_train,
            data_collator=data_collator,
            args=args,
        )
        seed_it(42)
        trainer.train()

    def get_response(self, sample, prompt_key='instruction'):
        prompt = ConRAGRunner.format_instruction_for_response(sample[prompt_key])
        inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=self.max_prompt_length,
                    add_special_tokens=False,
                ).to('cuda')

        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=100,
            do_sample=False,
            num_beams=self.beam_num if self.use_beam else 1,
            repetition_penalty=1.0,
            length_penalty=1,
            temperature=1.0,
        )[:, inputs.input_ids.shape[1]:]
        output_text = self.tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )
        output_text = [text.strip() for text in output_text]
        return output_text
    
    def eval(self):
        print('##############################  evaluation_from_list  ##############################')
        self.model.eval()
        res = []
        for data in tqdm(self.instruction_dataset_test[:], desc='get_response'):
            cur_res = self.get_response(data)[0]
            res.append(cur_res)
        if self.save_results:
            save_pkl_file = f'res_spring' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if not os.path.exists(f'output/{self.dataset_name}'):
                os.makedirs(f'output/{self.dataset_name}')
            pkl_save_path = f'output/{self.dataset_name}/{save_pkl_file}.pkl'
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
        self.load_dataset()
        self.load_model()
        self.start_training()
        new_embeddings = self.model.prompt_encoder.default.embedding.weight
        self.tokenizer, self.model = load_tokens(self.model.base_model, self.tokenizer, new_embeddings)
        print('new_embeddings', new_embeddings.shape)

        dataset_test = []
        for sample in self.instruction_dataset_test[:]:
            sample = deepcopy(sample)
            data = sample['instruction'].split('\n\n')
            reference = '\n\n'.join(data[:-1])
            question = data[-1]
            added_tokens = [f" [ref{i}]" for i in range(1, random.randint(1, 50))]
            # added_tokens = [f" [ref{i}]" for i in range(1, 51)]
            sample['instruction'] =  f"{reference}{added_tokens}\n\n{question}"
            dataset_test.append(sample)
        self.instruction_dataset_test = dataset_test
        self.eval()

def main(dataset_name, input_path, train_data_path, test_data_path, dataset_names, paths, **args):
    if dataset_name in ['hotpotqa', 'musique', '2wiki', 'bioasq']:
        input_path = {'train_data_path': train_data_path, 'test_data_path': test_data_path}
    print(args)
    runner = SpringRunner(
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

    parser.add_argument('--model_type', type=str, default='rag', choices=['spring'], help='model type')
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



