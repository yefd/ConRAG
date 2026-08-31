nohup python runner.py \
    --dataset_name 2wiki \
    --train_data_path dataset/2wikimultihop/2wikimultihop_train_bert_emb.pkl \
    --test_data_path dataset/2wikimultihop/2wikimultihop_dev_bert_emb.pkl \
    --standard_prompt \
    --model_name models/Llama-2-7b-chat-hf \
    --max_prompt_length 4096 \
    --output_dir output/models/conrag_7b_2wiki \
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
    >  logs/2wiki/output_conrag_llama2_e1.log 2>&1 ;

nohup python runner.py \
    --dataset_name 2wiki \
    --train_data_path dataset/2wikimultihop/2wikimultihop_train_bert_emb.pkl \
    --test_data_path dataset/2wikimultihop/2wikimultihop_dev_bert_emb.pkl \
    --standard_prompt \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --output_dir output/models/conrag_7b_2wiki \
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
    >  logs/2wiki/output_conrag_mistral_e1.log 2>&1 &