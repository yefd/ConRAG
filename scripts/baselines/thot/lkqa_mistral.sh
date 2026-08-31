


model_type="thot"
date="20250101"
prompt_type="qa_thot"

dataset_name="lkqa"

nohup python runner.py \
    --dataset_name $dataset_name \
    --input_path dataset/lkqa/lkqa_bge_emb.pkl \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --freeze_llm \
    --prompt_type $prompt_type \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/baselines/$model_type/output_${date}_${dataset_name}_mistral.log 2>&1 ;

nohup python runner.py \
    --dataset_name $dataset_name \
    --input_path dataset/lkqa/lkqa_bge_emb.pkl \
    --model_name models/Qwen2-7B-Instruct \
    --max_prompt_length 4096 \
    --freeze_llm \
    --prompt_type $prompt_type \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type qwen \
    >  logs/baselines/$model_type/output_${date}_${dataset_name}_qwen.log 2>&1 ;