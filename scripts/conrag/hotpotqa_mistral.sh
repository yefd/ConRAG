


nohup python runner.py \
    --dataset_name hotpotqa \
    --train_data_path dataset/hotpotqa/hotpot_train_v1.1_bert_emb.pkl \
    --test_data_path dataset/hotpotqa/hotpot_dev_distractor_v1_bert_emb.pkl \
    --standard_prompt \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --output_dir output/models/conrag_7b_hotpotqa_1118 \
    --use_training \
    --freeze_llm \
    --completion_only \
    --num_k 10 \
    --input_dim 768 \
    --model_type conrag \
    --use_evaluation \
    --save_results \
    --save_model \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/hotpotqa/output_conrag_standard_mistral.log 2>&1 &

nohup python runner.py \
    --dataset_name hotpotqa \
    --train_data_path dataset/hotpotqa/hotpot_train_v1.1_bert_emb.pkl \
    --test_data_path dataset/hotpotqa/hotpot_dev_distractor_v1_bert_emb.pkl \
    --standard_prompt \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --output_dir output/conrag_7b \
    --use_training \
    --freeze_llm \
    --completion_only \
    --num_k 10 \
    --input_dim 768 \
    --model_type conrag \
    --use_evaluation \
    --save_results \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/hotpotqa/output_conrag_standard_mistral_e2.log 2>&1 &