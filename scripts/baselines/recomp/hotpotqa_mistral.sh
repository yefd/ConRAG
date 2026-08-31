


model_type="recomp"
date="20250101"

dataset_name="hotpotqa"
nohup python runner.py \
    --dataset_name $dataset_name \
    --train_data_path no_path \
    --test_data_path output/recomp/hotpotqa_recomp_compress_gpt-4o-mini_single.jsonl \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/$model_type/output_${date}_${dataset_name}_mistral_single.log 2>&1 ;

nohup python runner.py \
    --dataset_name $dataset_name \
    --train_data_path no_path \
    --test_data_path output/recomp/hotpotqa_recomp_compress_gpt-4o-mini_single.jsonl \
    --model_name models/Llama-2-7b-chat-hf \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/$model_type/output_${date}_${dataset_name}_llama2_single.log 2>&1 ;