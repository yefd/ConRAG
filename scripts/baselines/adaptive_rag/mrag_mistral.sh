model_type="adaptive_rag"
date="20250101"
dataset_name="mrag"

nohup python baseline_adaptive_rag.py \
    --dataset_name $dataset_name \
    --input_path dataset/multihop_rag/multihop_rag_emb.pkl \
    --model_name models/longchat-7b-v1.5-32k \
    --classifier_name models/flan-t5-base \
    --max_prompt_length 32000 \
    --use_flash_att \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/baselines/$model_type/output_${date}_${dataset_name}_longchat.log 2>&1 ;

nohup python baseline_adaptive_rag.py \
    --dataset_name $dataset_name \
    --input_path dataset/multihop_rag/multihop_rag_emb.pkl \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --classifier_name models/flan-t5-base \
    --max_prompt_length 32000 \
    --use_flash_att \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/baselines/$model_type/output_${date}_${dataset_name}_mistral_flash.log 2>&1 ;