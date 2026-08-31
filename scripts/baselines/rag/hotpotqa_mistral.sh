nohup python runner.py \
    --dataset_name hotpotqa \
    --train_data_path dataset/hotpotqa/hotpot_train_v1.1_bert.pkl \
    --test_data_path dataset/hotpotqa/hotpot_dev_distractor_v1_bert.pkl \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/hotpotqa/output_rag_mistral.log 2>&1 &

nohup python runner.py \
    --dataset_name hotpotqa \
    --train_data_path dataset/hotpotqa/hotpot_train_v1.1_bert.pkl \
    --test_data_path dataset/hotpotqa/hotpot_dev_distractor_v1_bert.pkl \
    --model_name models/Llama-2-7b-chat-hf \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/hotpotqa/output_rag_llama2.log 2>&1 &

nohup python runner.py \
    --dataset_name hotpotqa \
    --train_data_path dataset/hotpotqa/hotpot_train_v1.1_bert.pkl \
    --test_data_path dataset/hotpotqa/hotpot_dev_distractor_v1_bert.pkl \
    --model_name models/Llama-2-13b-chat-hf \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/hotpotqa/output_rag_llama_13b.log 2>&1 &
