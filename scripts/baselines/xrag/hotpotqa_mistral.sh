


nohup python baseline_xrag.py \
    --dataset_name hotpotqa \
    --train_data_path dataset/hotpotqa/hotpot_train_v1.1_bert_emb.pkl \
    --test_data_path dataset/hotpotqa/hotpot_dev_distractor_v1_bert_emb.pkl \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --output_dir output/conrag_7b \
    --use_training \
    --freeze_llm \
    --num_k 10 \
    --input_dim 768 \
    --model_type xrag \
    --use_evaluation \
    --save_results \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/xrag/output_hotpotqa_mistral.log 2>&1 ;

nohup python baseline_xrag.py \
    --dataset_name hotpotqa \
    --train_data_path dataset/hotpotqa/hotpot_train_v1.1_bert_emb.pkl \
    --test_data_path dataset/hotpotqa/hotpot_dev_distractor_v1_bert_emb.pkl \
    --model_name models/Llama-2-7b-chat-hf \
    --max_prompt_length 4096 \
    --output_dir output/conrag_7b \
    --use_training \
    --freeze_llm \
    --num_k 10 \
    --input_dim 768 \
    --model_type xrag \
    --use_evaluation \
    --save_results \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/xrag/output_hotpotqa_llama2.log 2>&1 ;
