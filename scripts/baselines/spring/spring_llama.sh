


nohup python baseline_spring.py \
    --dataset_name nq_10 \
    --input_path dataset/nq/nq-open-10_total_documents_gold_bert_emb.pkl \
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
    >  logs/spring/output_nq10_llama2.log 2>&1 ;

nohup python baseline_spring.py \
    --dataset_name nq_20 \
    --input_path dataset/nq/nq-open-20_total_documents_gold_bert_emb.pkl \
    --model_name models/Llama-2-7b-chat-hf \
    --max_prompt_length 4096 \
    --output_dir output/conrag_7b \
    --use_training \
    --freeze_llm \
    --num_k 20 \
    --input_dim 768 \
    --model_type spring \
    --use_evaluation \
    --save_results \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/spring/output_nq20_llama2.log 2>&1 ;

nohup python baseline_spring.py \
    --dataset_name mrag \
    --input_path dataset/multihop_rag/multihop_rag_emb.pkl \
    --model_name models/longchat-7b-v1.5-32k \
    --max_prompt_length 32000 \
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
    >  logs/spring/output_mrag_llama2.log 2>&1 ;


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

nohup python baseline_spring.py \
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
    --model_type spring \
    --use_evaluation \
    --save_results \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --instruction_type chat \
    >  logs/spring/output_hotpotqa_llama2.log 2>&1 ;
