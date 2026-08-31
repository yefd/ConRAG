


dataset_name="hotpotqa"
model_type="prompt_tuning"
date="20250101"

nohup python runner.py \
    --dataset_name $dataset_name \
    --train_data_path dataset/hotpotqa/hotpot_train_v1.1_contriever_8k_emb.pkl \
    --test_data_path dataset/hotpotqa/hotpot_dev_distractor_v1_contriever_2k_emb.pkl \
    --standard_prompt \
    --retrieval_aware_type $model_type \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --output_dir output/models/conrag_7b_pt \
    --use_training \
    --freeze_llm \
    --completion_only \
    --num_k 10 \
    --input_dim 768 \
    --model_type $model_type \
    --use_evaluation \
    --save_results \
    --save_model \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/baselines/$model_type/output_${date}_${dataset_name}_standard_mistral_contriever.log 2>&1 &

nohup python runner.py \
    --dataset_name $dataset_name \
    --train_data_path dataset/hotpotqa/hotpot_train_v1.1_bert.pkl \
    --test_data_path dataset/hotpotqa/hotpot_dev_distractor_v1_bert.pkl \
    --standard_prompt \
    --retrieval_aware_type $model_type \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --output_dir output/models/conrag_7b_pt \
    --use_training \
    --freeze_llm \
    --completion_only \
    --num_k 10 \
    --input_dim 768 \
    --model_type $model_type \
    --use_evaluation \
    --save_results \
    --save_model \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/baselines/$model_type/output_${date}_${dataset_name}_standard_mistral.log 2>&1 &

nohup python runner.py \
    --dataset_name $dataset_name \
    --train_data_path dataset/hotpotqa/hotpot_train_v1.1_bert.pkl \
    --test_data_path dataset/hotpotqa/hotpot_dev_distractor_v1_bert.pkl \
    --standard_prompt \
    --retrieval_aware_type $model_type \
    --model_name models/Llama-2-7b-chat-hf \
    --max_prompt_length 4096 \
    --output_dir output/conrag_7b \
    --use_training \
    --freeze_llm \
    --completion_only \
    --num_k 10 \
    --input_dim 768 \
    --model_type $model_type \
    --use_evaluation \
    --save_results \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/baselines/$model_type/output_${date}_${dataset_name}_standard_llama2.log 2>&1 ;

