


dataset_name="musique"
model_type="prompt_tuning"
date="20250101"

nohup python runner.py \
    --dataset_name $dataset_name \
    --train_data_path dataset/musique/musique_ans_v1.0_train_bert.pkl \
    --test_data_path dataset/musique/musique_ans_v1.0_dev_bert.pkl \
    --standard_prompt \
    --retrieval_aware_type $model_type \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --output_dir output/conrag_7b \
    --use_training \
    --freeze_llm \
    --completion_only \
    --num_k 20 \
    --input_dim 768 \
    --model_type $model_type \
    --use_evaluation \
    --save_results \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/baselines/$model_type/output_${date}_${dataset_name}_standard_mistral.log 2>&1 ;

nohup python runner.py \
    --dataset_name $dataset_name \
    --train_data_path dataset/musique/musique_ans_v1.0_train_bert.pkl \
    --test_data_path dataset/musique/musique_ans_v1.0_dev_bert.pkl \
    --standard_prompt \
    --retrieval_aware_type $model_type \
    --model_name models/Llama-2-7b-chat-hf \
    --max_prompt_length 4096 \
    --output_dir output/conrag_7b \
    --use_training \
    --freeze_llm \
    --completion_only \
    --num_k 20 \
    --input_dim 768 \
    --model_type $model_type \
    --use_evaluation \
    --save_results \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/baselines/$model_type/output_${date}_${dataset_name}_standard_llama2.log 2>&1 ;

