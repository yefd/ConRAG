


model_type="trace"
date="20250101"
dataset_name="mrag"

nohup python baseline_trace.py \
    --dataset_name $dataset_name \
    --input_path dataset/multihop_rag/multihop_rag_emb.pkl \
    --model_name models/longchat-7b-v1.5-32k \
    --reasoning_model_name models/Meta-Llama-3.1-8B-Instruct \
    --max_prompt_length 32000 \
    --use_flash_att \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/baselines/$model_type/output_${date}_${dataset_name}_longchat.log 2>&1 ;

nohup python baseline_trace.py \
    --dataset_name $dataset_name \
    --input_path dataset/multihop_rag/multihop_rag_emb.pkl \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --reasoning_model_name models/Meta-Llama-3.1-8B-Instruct \
    --max_prompt_length 32000 \
    --use_flash_att \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/baselines/$model_type/output_${date}_${dataset_name}_mistral_flash.log 2>&1 ;