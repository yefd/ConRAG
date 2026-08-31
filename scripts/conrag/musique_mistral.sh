


nohup python runner.py \
    --dataset_name musique \
    --train_data_path dataset/musique/musique_ans_v1.0_train_bert_emb.pkl \
    --test_data_path dataset/musique/musique_ans_v1.0_dev_bert_emb.pkl \
    --standard_prompt \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 32000 \
    --output_dir output/models/conrag_7b_musique_1117 \
    --use_training \
    --freeze_llm \
    --completion_only \
    --num_k 20 \
    --input_dim 768 \
    --model_type conrag \
    --use_evaluation \
    --save_results \
    --save_model \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/musique/output_conrag_standard_mistral_2.log 2>&1 &