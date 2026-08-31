


model_type="recomp"
date="20250101"

dataset_name="mrag"
nohup python runner.py \
    --dataset_name $dataset_name \
    --input_path output/recomp/mrag_recomp_compress_gpt-4o-mini_single.jsonl  \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 32000 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/$model_type/output_${date}_${dataset_name}_mistral_single.log 2>&1 ;

dataset_name="mrag"
nohup python runner.py \
    --dataset_name $dataset_name \
    --input_path output/recomp/mrag_recomp_compress_gpt-4o-mini_single.jsonl  \
    --model_name models/longchat-7b-v1.5-32k \
    --max_prompt_length 32000 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/$model_type/output_${date}_${dataset_name}_longchat_single.log 2>&1 ;