


model_type="prompt_tuning"
date="20250101"

dataset_name="bioasq"
nohup python runner.py \
    --dataset_name bioasq \
    --train_data_path dataset/bioasq/bioasq_train.pkl \
    --test_data_path dataset/bioasq/bioasq_test.pkl \
    --retrieval_aware_type $model_type \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 8192 \
    --output_dir output/conrag_7b \
    --use_training \
    --freeze_llm \
    --completion_only \
    --num_k 10 \
    --input_dim 1024 \
    --model_type $model_type \
    --use_evaluation \
    --save_results \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/baselines/$model_type/output_${date}_${dataset_name}_mistral.log 2>&1 &

dataset_name="lkqa"
nohup python runner.py \
    --dataset_name $dataset_name \
    --input_path dataset/lkqa/lkqa_bge_emb.pkl \
    --model_name models/Qwen2-7B-Instruct \
    --retrieval_aware_type $model_type \
    --max_prompt_length 8192 \
    --output_dir output/conrag_7b \
    --use_training \
    --freeze_llm \
    --completion_only \
    --num_k 10 \
    --input_dim 1024 \
    --hidden_size 3584 \
    --model_type $model_type \
    --use_evaluation \
    --save_results \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --instruction_type qwen \
    >  logs/baselines/$model_type/output_${date}_${dataset_name}_qwen2.log 2>&1 &


