import torch
import numpy as np
import random, os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(torch.cuda.is_available())
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
seed_it(42)


from tqdm import tqdm
import pickle
import argparse
from datetime import datetime
from transformers import AutoTokenizer
from datasets import Dataset
from transformers import TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, TaskType
from trl import DataCollatorForCompletionOnlyLM

from ConRAG.dataset.load_nq import load_nq_dataset, get_nq_ans
from ConRAG.dataset.load_hotpotqa import load_hotpotqa_dataset, get_hotpotqa_ans
from ConRAG.dataset.load_musique import load_musique_dataset, get_musique_ans
from ConRAG.dataset.load_dureader import load_dureader_dataset, get_dureader_ans
from ConRAG.dataset.load_mrag import load_mrag_dataset, get_mrag_ans
from ConRAG.dataset.load_bioasq import load_bioasq_dataset, get_bioasq_ans
from ConRAG.dataset.load_lkqa import load_lkqa_dataset, get_lkqa_ans
from ConRAG.models.modeling_conrag import ConRAGLlamaForCausalLM, ConRAGLlamaConfig
from ConRAG.models.modeling_rag import RAGLlamaForCausalLM, RAGLlamaConfig
from ConRAG.utils.trainer import ConRAGTrainer
from ConRAG.utils.metrics import evaluation_from_list

class ConRAGRunner:
    SEMANTIC_TOKEN = '<S>'
    UNK_TOKEN = '<unk>'
    UNK_TOKEN_ID = 0
    instruction_type = 'chat'

    def __init__(
        self,
        dataset_name='nq_10',
        input_path='',
        use_emb=True,
        max_prompt_length=4096,
        standard_prompt=True,
        retrieval_aware_type='',

        model_name='',
        load_in_8bit=False,
        use_flash_att=False,
        save_model=False,
        output_dir='',

        model_type='',
        input_dim=768,
        hidden_size=4096,
        SEMANTIC_TOKEN='<S>',
        UNK_TOKEN='<unk>',
        UNK_TOKEN_ID = 0,

        num_k=10,
        use_lora=False,
        use_cot=False,
        prompt_type='',
        use_training=False,
        freeze_llm=True,
        load_from_pretrained=False,
        pretrained_model_name='',
        num_train_epochs=1,
        per_device_train_batch_size=1,

        use_evaluation=True,
        max_new_tokens=100,
        use_beam=False,
        beam_num=5,
        save_results=False,
        instruction_type='chat',

        completion_only=True,
    ):
        self.dataset_name = dataset_name # 
        self.input_path = input_path
        self.use_emb = model_type in ['conrag'] or ('emb' in str(self.input_path))
        self.max_prompt_length = max_prompt_length
        self.standard_prompt = standard_prompt
        self.retrieval_aware_type = retrieval_aware_type

        self.model_name = model_name
        self.load_in_8bit = load_in_8bit
        self.use_flash_att = use_flash_att
        self.save_model = save_model
        self.output_dir = output_dir

        self.model_type = model_type
        self.retrieval_aware = model_type in ['conrag']
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        ConRAGRunner.set_retrieval_token(SEMANTIC_TOKEN)
        ConRAGRunner.set_unk_token(UNK_TOKEN)
        ConRAGRunner.set_unk_token_id(UNK_TOKEN_ID)

        self.num_k = num_k
        self.use_training = use_training
        self.freeze_llm = freeze_llm
        self.load_from_pretrained = load_from_pretrained
        self.pretrained_model_name = pretrained_model_name
        self.use_lora = use_lora
        self.use_cot = use_cot
        self.prompt_type = prompt_type
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.completion_only = completion_only

        self.use_evaluation = use_evaluation
        self.max_new_tokens = max_new_tokens
        self.use_beam = use_beam
        self.beam_num = beam_num
        self.save_results = save_results

        ConRAGRunner.set_instruction_type(instruction_type)

    @classmethod
    def set_unk_token(cls, token):
        cls.UNK_TOKEN = token

    @classmethod
    def set_retrieval_token(cls, token):
        cls.SEMANTIC_TOKEN = token

    @classmethod
    def set_unk_token_id(cls, id):
        cls.UNK_TOKEN_ID = id

    @classmethod
    def set_instruction_type(cls, instruction_type):
        cls.instruction_type = instruction_type

    @staticmethod
    def format_instruction(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            if ConRAGRunner.instruction_type == 'instruction':
                text = "### Instruction: \n{instruction}\n\n### Response:\n{output}\n".format(
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
    def format_instruction_for_response(prompt):
        if ConRAGRunner.instruction_type == 'instruction':
            text = "### Instruction: \n{instruction}\n\n### Response:\n".format(
                instruction=prompt,
                )
        elif ConRAGRunner.instruction_type == 'chat':
            text = "[INST] {instruction} [/INST]".format(
                instruction=prompt,
                )
        elif ConRAGRunner.instruction_type == 'qwen':
            text = "<|im_start|>user\n{instruction}<|im_end|>".format(
                instruction=prompt,
                )
        text = text.replace(ConRAGRunner.SEMANTIC_TOKEN, ConRAGRunner.UNK_TOKEN)
        return text

    def load_tokenizer(self):
        if 'Qwen' in self.model_name: # Qwen
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token=True, trust_remote_code=True)
            tokenizer.padding_side = "left"
            tokenizer.unk_token = '<|endoftext|>'
            tokenizer.pad_token = tokenizer.eos_token
            ConRAGRunner.set_unk_token(tokenizer.unk_token)
            ConRAGRunner.set_unk_token_id(tokenizer.unk_token_id)
        else: # Llama
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token=True)
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

    def load_dataset(self, ignore_train=False):
        self.load_tokenizer()
        if self.dataset_name not in ['nq_10', 'nq_20', 'hotpotqa', 'musique', '2wiki', 'dureader', 'mrag', 'bioasq', 'lkqa']:
            raise ValueError(self.dataset_name)
        if 'nq' in self.dataset_name:
            _load_dataset = load_nq_dataset
        elif self.dataset_name == 'hotpotqa' or self.dataset_name == '2wiki':
            _load_dataset = load_hotpotqa_dataset
        elif self.dataset_name == 'musique':
            _load_dataset = load_musique_dataset
        elif self.dataset_name == 'mrag':
            _load_dataset = load_mrag_dataset
        elif self.dataset_name == 'bioasq':
            _load_dataset = load_bioasq_dataset
        elif self.dataset_name == 'lkqa':
            _load_dataset = load_lkqa_dataset
        elif self.dataset_name == 'dureader':
            _load_dataset = load_dureader_dataset
        self.instruction_dataset_train, self.instruction_dataset_test = _load_dataset(
            input_path=self.input_path, 
            max_prompt_length=self.max_prompt_length, 
            tokenizer=self.tokenizer, 
            retrieval_aware=self.retrieval_aware, 
            use_cot=self.use_cot, 
            prompt_type=self.prompt_type,
            SEMANTIC_TOKEN=self.SEMANTIC_TOKEN,
            use_emb=self.use_emb,
            standard_prompt=self.standard_prompt,
            retrieval_aware_type=self.retrieval_aware_type,
            num_k=self.num_k,
            ignore_train=ignore_train,
            )
        print('self.instruction_dataset_train[0]', self.instruction_dataset_train[0] if (not ignore_train and self.instruction_dataset_train != None) else None)
        print('self.instruction_dataset_test[0]', self.instruction_dataset_test[0])
    
    def load_model(self):
        print('##############################  load_model  ##############################')
        if self.model_type == 'conrag':
            if self.load_from_pretrained:
                print(f'load_from_pretrained: {self.pretrained_model_name}')
                config = ConRAGLlamaConfig.from_pretrained(self.pretrained_model_name)
                self.model = ConRAGLlamaForCausalLM.from_pretrained(self.pretrained_model_name, config=config)
            else:
                config = ConRAGLlamaConfig(
                    model_name_or_path=self.model_name,
                    load_in_8bit=self.load_in_8bit,
                    input_dim=self.input_dim,
                    hidden_size=self.hidden_size,
                    unk_token=self.UNK_TOKEN,
                    unk_token_id=self.UNK_TOKEN_ID,
                    freeze_llm=self.freeze_llm,
                    num_k=self.num_k,
                    use_flash_att=self.use_flash_att,
                    )
                self.model = ConRAGLlamaForCausalLM(config)
        else:
            config = RAGLlamaConfig(
                model_name_or_path=self.model_name,
                load_in_8bit=self.load_in_8bit,
                freeze_llm=self.freeze_llm,
                use_flash_att=self.use_flash_att,
                )
            self.model = RAGLlamaForCausalLM(config)
        print(config)
        print(self.model)
    
    def get_peft_model(self):
        peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                bias="none",
                inference_mode=False,
                task_type=TaskType.CAUSAL_LM, 
                target_modules=["q_proj", "v_proj"]
        )
        print(peft_config)
        self.model.llama_model = prepare_model_for_kbit_training(self.model.llama_model)
        self.model.llama_model = get_peft_model(self.model.llama_model, peft_config)
        self.model.llama_model.print_trainable_parameters()
        return peft_config
    
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
            learning_rate=2e-4,
            bf16=True,
            tf32=True,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
            # disable_tqdm=True # disable tqdm since with packing values are in correct
        )
        print(args)
        dataset_train = Dataset.from_list(self.instruction_dataset_train[:])
        peft_config = self.get_peft_model() if self.use_lora else None
        max_seq_length = self.max_prompt_length
        response_tag = 2
        data_collator = DataCollatorForCompletionOnlyLM(self.tokenizer.encode("\nAnswer:", add_special_tokens=False)[response_tag:], tokenizer=self.tokenizer) if self.completion_only else None
        trainer = ConRAGTrainer(
            model=self.model,
            train_dataset=dataset_train,
            peft_config=peft_config,
            max_seq_length=max_seq_length,
            tokenizer=self.tokenizer,
            packing=False,
            formatting_func=ConRAGRunner.format_instruction,
            data_collator= data_collator,
            args=args,
        )
        seed_it(42)
        trainer.train()
        # save model
        if self.save_model:
            self.output_dir = f'{self.output_dir}_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            print('save model to:', self.output_dir)
            self.model.save_model(self.output_dir)
        else:
            print('dont save_model')

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
        if self.model_type in ['conrag', 'prompt_tuning']:
            embeds = torch.tensor(sample['embeds']).to(input_tokens.input_ids.device)
            label = torch.tensor(sample['label']).to(input_tokens.input_ids.device)
            if embeds.dim() == 1:
                embeds = embeds.unsqueeze(0)
            if label.dim() == 1:
                label = label.unsqueeze(0)
            inputs = {"input_ids": input_tokens['input_ids'], 'attention_mask': input_tokens['attention_mask'], 'embeds': embeds, 'label': label}
        else:
            inputs = {"input_ids": input_tokens['input_ids'], 'attention_mask': input_tokens['attention_mask']}
        max_new_tokens = 100 if 'thot' not in self.prompt_type else 1024
        outputs = self.model.generate(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=self.beam_num if self.use_beam else 1,
            repetition_penalty=1.0,
            length_penalty=1,
            temperature=1.0,
            pad_token_id=self.tokenizer.eos_token_id
        )
        output_text = self.tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )
        output_text = [text.strip() for text in output_text]
        return output_text

    def get_response_thot(self, sample, output_text, prompt_key='instruction'):
        prompt = sample[prompt_key]
        prompt = prompt + output_text + 'Therefore, the answer: \n'
        prompt = ConRAGRunner.format_instruction_for_response(prompt)
        input_tokens = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=self.max_prompt_length,
                    add_special_tokens=False,
                ).to('cuda')
        inputs = {"input_ids": input_tokens['input_ids'], 'attention_mask': input_tokens['attention_mask']}
        outputs = self.model.generate(
            inputs=inputs,
            max_new_tokens=100,
            do_sample=False,
            num_beams=self.beam_num if self.use_beam else 1,
            repetition_penalty=1.0,
            length_penalty=1,
            temperature=1.0,
            pad_token_id=self.tokenizer.eos_token_id
        )
        output_text = self.tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )
        output_text = [text.strip() for text in output_text]
        return output_text

    def eval(self):
        print('##############################  evaluation_from_list  ##############################')
        self.model.eval()
        self.model.llama_model.eval()
        res = []
        for data in tqdm(self.instruction_dataset_test[:], desc=f'get_response {self.prompt_type}'):
            cur_res = self.get_response(data)[0]
            if 'thot' in self.prompt_type:
                cur_res = self.get_response_thot(data, cur_res)[0]
            res.append(cur_res)
        if self.save_results:
            save_pkl_file = f'res_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
        print(self.use_training, self.use_evaluation)
        if self.use_training:
            self.start_training()
        if self.use_evaluation:
            self.eval()
    
    def run1eval2(self, dataset_names, paths):
        self.load_dataset()
        self.load_model()
        print(self.use_training, self.use_evaluation)
        if self.use_training:
            self.start_training()
        print(f'eval {self.dataset_name}')
        if self.use_evaluation:
            self.eval()
        for dataset_name in dataset_names:
            if dataset_name in ['hotpotqa', 'musique', '2wiki', 'bioasq']:
                input_path = {'train_data_path': paths[0], 'test_data_path': paths[1]}
                paths = paths[2:]
            else:
                input_path = paths[0]
                paths = paths[1:]
            self.dataset_name = dataset_name
            self.input_path = input_path
            print(f'eval {dataset_name}')
            self.load_dataset(ignore_train=True)
            self.eval()

def main(dataset_name, input_path, train_data_path, test_data_path, dataset_names, paths, **args):
    if len(dataset_names) == 0:
        if dataset_name in ['hotpotqa', 'musique', '2wiki', 'bioasq']:
            input_path = {'train_data_path': train_data_path, 'test_data_path': test_data_path}
        print(args)
        runner = ConRAGRunner(
            dataset_name=dataset_name,
            input_path=input_path, 
            **args
            )
        print('runner', runner.__dict__)
        runner.run()
    else:
        if dataset_names[0] in ['hotpotqa', 'musique', '2wiki', 'bioasq']:
            input_path = {'train_data_path': paths[0], 'test_data_path': paths[1]}
            paths = paths[2:]
        else:
            input_path = paths[0]
            paths = paths[1:]
        print(dataset_names[0], input_path)
        print(args)
        runner = ConRAGRunner(
            dataset_name=dataset_names[0],
            input_path=input_path, 
            **args
            )
        print('runner', runner.__dict__)
        runner.run1eval2(dataset_names[1:], paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model with given parameters")
    
    parser.add_argument('--dataset_name', type=str, default='', choices=['nq_10', 'nq_20', 'hotpotqa', 'musique', '2wiki', 'dureader', 'mrag', 'bioasq', 'lkqa'], help='Name of the dataset')
    parser.add_argument('--input_path', type=str, default=None, help='Path for nq or dureader datasets')
    parser.add_argument('--train_data_path', type=str, default=None, help='Path for the hotpotqa, musique, 2wiki')
    parser.add_argument('--test_data_path', type=str, default=None, help='Path for the hotpotqa, musique, 2wiki')
    parser.add_argument('--dataset_names', nargs='+',type=str, default=[], help='Name of the dataset')
    parser.add_argument('--paths', nargs='+', type=str, default=[], help='Paths for datasets')

    parser.add_argument('--max_prompt_length', type=int, default=4096, help='Maximum prompt length')
    parser.add_argument('--standard_prompt', action='store_true', help='Use standard prompt')
    parser.add_argument('--retrieval_aware_type', type=str, default='', help='Retrieval aware prompting type')

    parser.add_argument('--model_name', type=str, required=True, help='Name of LLM')
    parser.add_argument('--load_in_8bit', action='store_true', help='Load in 8-bit precision')
    parser.add_argument('--use_flash_att', action='store_true', help='Load in use_flash_att')
    parser.add_argument('--save_model', action='store_true', help='If set, the trained model will be saved to outputdir')
    parser.add_argument('--output_dir', type=str, required=False, help='Directory to save model')

    parser.add_argument('--model_type', type=str, default='rag', choices=['rag', 'conrag', 'prompt_tuning'], help='model type')
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
    parser.add_argument('--prompt_type', type=str, default='', help='Type of prompt template')
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
