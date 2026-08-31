
conda activate nl 

model_type="recomp"
date="20250101"
dataset_name="nq_20"
nohup python runner.py \
    --dataset_name $dataset_name \
    --input_path output/recomp/nq20_recomp_compress_gpt-4o-mini_top10.jsonl \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 20 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/baselines/$model_type/output_${date}_${dataset_name}_mistral.log 2>&1 ;

nohup python runner.py \
    --dataset_name $dataset_name \
    --input_path output/recomp/nq20_recomp_compress_gpt-4o-mini.jsonl \
    --model_name models/Llama-2-7b-chat-hf \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 20 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/baselines/$model_type/output_${date}_${dataset_name}_llama2.log 2>&1 ;
