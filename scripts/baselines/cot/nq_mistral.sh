model_type="cot"
date="20250101"

dataset_name="nq_10"
nohup python runner.py \
    --dataset_name $dataset_name \
    --input_path dataset/nq/nq-open-10_total_documents_gold_bert.pkl \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --freeze_llm \
    --use_cot \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/$model_type/output_${date}_${dataset_name}_rag_mistral.log 2>&1 ;

nohup python runner.py \
    --dataset_name $dataset_name \
    --input_path dataset/nq/nq-open-10_total_documents_gold_bert.pkl \
    --model_name models/Llama-2-7b-chat-hf \
    --max_prompt_length 4096 \
    --freeze_llm \
    --use_cot \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/$model_type/output_${date}_${dataset_name}_rag_llama2.log 2>&1 ;

dataset_name="nq_20"
nohup python runner.py \
    --dataset_name $dataset_name \
    --input_path dataset/nq/nq-open-20_total_documents_gold_bert.pkl \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 8192 \
    --freeze_llm \
    --use_cot \
    --num_k 20 \
    --input_dim 768 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/$model_type/output_${date}_${dataset_name}_rag_mistral.log 2>&1 ;

nohup python runner.py \
    --dataset_name $dataset_name \
    --input_path dataset/nq/nq-open-20_total_documents_gold_bert.pkl \
    --model_name models/Llama-2-7b-chat-hf \
    --max_prompt_length 4096 \
    --freeze_llm \
    --use_cot \
    --num_k 20 \
    --input_dim 768 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/$model_type/output_${date}_${dataset_name}_rag_llama2.log 2>&1 ;