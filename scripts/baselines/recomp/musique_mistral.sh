
conda activate nl 

model_type="recomp"
date="20250101"
dataset_name="musique"

nohup python runner.py \
    --dataset_name $dataset_name \
    --train_data_path no_path \
    --test_data_path output/recomp/musique_recomp_compress_gpt-3.5_single_top10.jsonl \
    --model_name models/Llama-2-7b-chat-hf \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 20 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/$model_type/output_${date}_${dataset_name}_llama2_single_3.5_top10.log 2>&1 ;

nohup python runner.py \
    --dataset_name $dataset_name \
    --train_data_path no_path \
    --test_data_path output/recomp/musique_recomp_compress_gpt-3.5_single_top10.jsonl \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 20 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/$model_type/output_${date}_${dataset_name}_mistral_single_3.5_top10.log 2>&1 ;