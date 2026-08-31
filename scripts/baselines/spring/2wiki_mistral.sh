


nohup python baseline_spring.py \
    --dataset_name 2wiki \
    --train_data_path dataset/2wikimultihop/2wikimultihop_train_bert_emb.pkl \
    --test_data_path dataset/2wikimultihop/2wikimultihop_dev_bert_emb.pkl \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --output_dir output/conrag_7b \
    --use_training \
    --freeze_llm \
    --num_k 10 \
    --input_dim 768 \
    --model_type spring \
    --use_evaluation \
    --save_results \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/spring/output_2wiki_mistral.log 2>&1 ;

nohup python baseline_spring.py \
    --dataset_name 2wiki \
    --train_data_path dataset/2wikimultihop/2wikimultihop_train_bert_emb.pkl \
    --test_data_path dataset/2wikimultihop/2wikimultihop_dev_bert_emb.pkl \
    --model_name models/Llama-2-7b-chat-hf \
    --max_prompt_length 4096 \
    --output_dir output/conrag_7b \
    --use_training \
    --freeze_llm \
    --num_k 10 \
    --input_dim 768 \
    --model_type spring \
    --use_evaluation \
    --save_results \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/spring/output_2wiki_llama2.log 2>&1 ;