nohup python runner.py \
    --dataset_name musique \
    --train_data_path dataset/musique/musique_ans_v1.0_train_bert.pkl \
    --test_data_path dataset/musique/musique_ans_v1.0_dev_bert.pkl \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 20 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/musique/output_rag_mistral.log 2>&1 &



nohup python runner.py \
    --dataset_name musique \
    --train_data_path dataset/musique/musique_ans_v1.0_train_bert.pkl \
    --test_data_path dataset/musique/musique_ans_v1.0_dev_bert.pkl \
    --model_name models/Llama-2-7b-chat-hf \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 20 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/musique/output_rag_llama2.log 2>&1 &

nohup python runner.py \
    --dataset_name musique \
    --train_data_path dataset/musique/musique_ans_v1.0_train_bert.pkl \
    --test_data_path dataset/musique/musique_ans_v1.0_dev_bert.pkl \
    --model_name models/Llama-2-13b-chat-hf \
    --max_prompt_length 4096 \
    --freeze_llm \
    --num_k 20 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/musique/output_rag_llama_13b.log 2>&1 &