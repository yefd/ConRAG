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
from ConRAG.utils.metrics import *

def evaluation_from_sample(responses, answers, dataset_name):
    # if 'nq' in dataset_name:
    #     METRICS = METRICS_NQ
    if 'dureader' in dataset_name or 'lkqa' in dataset_name:
        METRICS = METRICS_ZH
    elif 'bioasq' in dataset_name:
        METRICS = METRICS_BIO
    else:
        METRICS = METRICS_EN

    all_example_metrics = []
    if len(responses) != len(answers):
        raise ValueError
    for i in range(len(responses)):
        all_example_metrics.append(get_metrics_for_example({'model_answer': responses[i], 'answers': answers[i]}, METRICS))

    # Average metrics across examples

    for (_, metric_name) in METRICS:
        average_metric_value = statistics.mean(
            example_metrics[metric_name] for (example_metrics, _) in all_example_metrics
        )
        logger.info(f"{metric_name}: {average_metric_value}")
    return all_example_metrics

   
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from tqdm import trange
import numpy as np

class AdaptiveJudger:
    """Implementation for Adaptive-RAG (two classification version)
    Paper link: https://aclanthology.org/2024.naacl-long.389.pdf
    """
    def __init__(self, model_path, batch_size=16, max_length=512):
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_length = max_length
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model.cuda()
        self.model.eval()

    def train(self, train_dataset, learning_rate=3e-5, epochs=5):
        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            save_strategy="no",
            logging_dir="./logs",
            logging_steps=100,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )

        trainer.train()


    @torch.inference_mode()
    def predict(self, test_dataset):
        q_key= 'question'
        questions = [sample[q_key] for sample in test_dataset]
        all_preds = []

        for idx in trange(0, len(questions), self.batch_size, desc="Prediction process: "):
            batch_input = questions[idx : idx + self.batch_size]
            batch_input = self.tokenizer(
                batch_input,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.model.device)

            scores = self.model.generate(
                **batch_input, return_dict_in_generate=True, output_scores=True, max_length=2
            ).scores[0]

            probs = (
                torch.nn.functional.softmax(
                    torch.stack(
                        [
                            scores[:, self.tokenizer("0").input_ids[0]],
                            scores[:, self.tokenizer("1").input_ids[0]],
                        ]
                    ),
                    dim=0,
                )
                .detach()
                .cpu()
                .numpy()
            )

            preds_labels = np.argmax(probs, 0)
            all_preds.extend(preds_labels)

        return all_preds


class ClassificationDataset(Dataset):
    def __init__(self, dataset_name, data, tokenizer, max_length=512):
        self.dataset_name = dataset_name
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q_key= 'query' if self.dataset_name == 'mrag' else 'question'
        question = self.data[idx][q_key]
        label = self.data[idx]["silver_label"]
        label_text = str(label)

        inputs = self.tokenizer(
            question,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = self.tokenizer(
                label_text,
                max_length=2,
                padding="max_length",
                truncation=True,
            )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        labels = torch.tensor(labels["input_ids"]).squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
from runner import *
from transformers.utils.logging import disable_progress_bar
disable_progress_bar()

class AdaptiveRAGRunner(ConRAGRunner):
    def __init__(
        self,
        classifier_name='flan-t5-base',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.classifier_name = classifier_name
        self.use_emb = False
    

    def get_question_response(self, sample,):
        prompt_template = "Question: {question}\nAnswer:"
        q_key= 'query' if self.dataset_name == 'mrag' else 'question'
        prompt = prompt_template.format(question=sample[q_key])
        prompt = ConRAGRunner.format_instruction_for_response(prompt)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_prompt_length,
            add_special_tokens=False,
        ).to('cuda')
        with torch.no_grad():
            embed_tokens = self.model.llama_model.get_input_embeddings()
            inputs_embeds = embed_tokens(inputs.input_ids)
            outputs = self.model.llama_model.generate(inputs_embeds=inputs_embeds,
                                            attention_mask=inputs.attention_mask,
                                            max_new_tokens=100,
                                            do_sample=False,
                                            repetition_penalty=1.0,
                                            length_penalty=1,
                                            temperature=1.0,
                                            pad_token_id=self.tokenizer.eos_token_id,
                                            )
        
        # Decode and return output text
        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]
        return output_text

    def _get_silver_dataset(self, dataset):
        print('##############################  evaluation_from_list  ##############################')
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
        self.get_ans = get_ans
        self.model.llama_model.eval()
        
        processed_data = []
        
        for sample in tqdm(dataset, desc='_get_silver_dataset'):
            # Get the model's response to the question in the prompt template
            response = self.get_question_response(
                sample=sample,
            )[0]
            gt_ans = get_ans([sample])
            m = evaluation_from_sample([response], gt_ans, self.dataset_name)
            is_correct = m[0][0]['best_subspan_em']
            sample["silver_label"] = 1 if is_correct else 0
            sample["silver_answer"] = response
            processed_data.append(sample)
        print('Num of simple questions: ', sum([d['silver_label'] for d in dataset]))
        print('percentage: ', sum([d['silver_label'] for d in dataset]) / len(dataset))
        return processed_data
    
    def get_silver_dataset(self, ):
        self.instruction_dataset_train = self._get_silver_dataset(self.instruction_dataset_train)
        self.instruction_dataset_test = self._get_silver_dataset(self.instruction_dataset_test)


    def run_classifier(self, ):
        cls_tokenizer = AutoTokenizer.from_pretrained(self.classifier_name)
        train_dataset = ClassificationDataset(self.dataset_name, self.instruction_dataset_train, cls_tokenizer)
        judger = AdaptiveJudger(model_path=self.classifier_name, batch_size=4, max_length=512)
        judger.train(train_dataset,)
        predicted_labels = judger.predict(self.instruction_dataset_test)
        for i, sample in enumerate(self.instruction_dataset_test):
            sample["predicted_label"] = predicted_labels[i]
        from sklearn.metrics import accuracy_score as acc_score_fn
        from sklearn.metrics import f1_score as f1_score_fn

        silver_labels = [sample["silver_label"] for sample in self.instruction_dataset_test]
        accuracy = acc_score_fn(silver_labels, predicted_labels)
        f1 = f1_score_fn(silver_labels, predicted_labels)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
    

    def get_adaptive_response(self, sample):
        tokenizer = self.tokenizer
        model = self.model.llama_model
        max_prompt_length = self.max_prompt_length
        if sample['predicted_label'] == 1:  ## simple question
            prompt_template = "Question: {question}\nAnswer:"
            q_key= 'query' if self.dataset_name == 'mrag' else 'question'
            prompt = prompt_template.format(question=sample[q_key])
        else:
            prompt = sample['instruction']
        prompt = ConRAGRunner.format_instruction_for_response(prompt)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_prompt_length,
            add_special_tokens=False,
        ).to('cuda')
        with torch.no_grad():
            embed_tokens = model.get_input_embeddings()
            inputs_embeds = embed_tokens(inputs.input_ids)
            outputs = model.generate(inputs_embeds=inputs_embeds,
                                            attention_mask=inputs.attention_mask,
                                            max_new_tokens=100,
                                            do_sample=False,
                                            repetition_penalty=1.0,
                                            length_penalty=1,
                                            temperature=1.0,
                                            pad_token_id=tokenizer.eos_token_id,
                                            )
        
        # Decode and return output text
        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]
        return output_text

    def eval_silver_dataset(self):
        dataset_name = self.dataset_name
        dataset = self.instruction_dataset_test
        from datetime import datetime
        from ConRAG.utils.metrics import evaluation_from_list
        print('##############################  evaluation_from_list  ##############################')
        self.model.eval()
        res = []
        for sample in tqdm(dataset, desc='get_response'):
            # Get the model's response to the question in the prompt template
            response = self.get_adaptive_response(
                sample=sample,
            )[0]
            res.append(response)
        if self.save_results:
            save_pkl_file = f'res_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            pkl_save_path = f'output/{dataset_name}/{save_pkl_file}.pkl'
            print('results_save_path', pkl_save_path)
            with open(pkl_save_path, 'wb') as f:
                pickle.dump(res, f)
                f.close()
        gt_ans = self.get_ans(dataset)
        m = evaluation_from_list(res[:], gt_ans[:len(res)], dataset_name)
        # print(m)
    
    def run(self):
        self.load_dataset()
        self.load_model()
        self.get_silver_dataset()
        self.run_classifier()
        self.eval_silver_dataset()

def main(dataset_name, input_path, train_data_path, test_data_path, dataset_names, paths, **args):
    if dataset_name in ['hotpotqa', 'musique', '2wiki', 'bioasq']:
        input_path = {'train_data_path': train_data_path, 'test_data_path': test_data_path}
    print(args)
    runner = AdaptiveRAGRunner(
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
    parser.add_argument('--classifier_name', type=str, required=True, help='Name of classifier')
    parser.add_argument('--freeze_llm', action='store_true', help='Freeze LLM')
    parser.add_argument('--input_dim', type=int, default=3, help='Input features')

    parser.add_argument('--load_in_8bit', action='store_true', help='Load in 8-bit precision')
    parser.add_argument('--use_flash_att', action='store_true', help='Load in use_flash_att')
    parser.add_argument('--model_type', type=str, default='rag', choices=['rag'], help='model type')
    parser.add_argument('--num_k', type=int, default=10, help='Number of retrieved documents')
    parser.add_argument('--use_evaluation', action='store_true', help='Use for evaluation')

    parser.add_argument('--save_results', action='store_true', help='Save results')
    parser.add_argument('--instruction_type', default='llama', choices=['chat', 'instruction', 'qwen'], help='instruction_type, llama or mistral')

    args = parser.parse_args()
    main(**vars(args))





