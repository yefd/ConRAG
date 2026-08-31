


model_type="trace"
date="20250101"

dataset_name="2wiki"

nohup python baseline_trace.py \
    --dataset_name 2wiki \
    --train_data_path dataset/2wikimultihop/2wikimultihop_train_bert_emb.pkl \
    --test_data_path dataset/2wikimultihop/2wikimultihop_dev_bert_emb.pkl \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --reasoning_model_name models/Meta-Llama-3.1-8B-Instruct \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/baselines/$model_type/output_${date}_${dataset_name}_rag_mistral.log 2>&1 ;

nohup python baseline_trace.py \
    --dataset_name 2wiki \
    --train_data_path dataset/2wikimultihop/2wikimultihop_train_bert_emb.pkl \
    --test_data_path dataset/2wikimultihop/2wikimultihop_dev_bert_emb.pkl \
    --model_name models/Llama-2-7b-chat-hf \
    --reasoning_model_name models/Meta-Llama-3.1-8B-Instruct \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/baselines/$model_type/output_${date}_${dataset_name}_rag_llama2.log 2>&1 ;