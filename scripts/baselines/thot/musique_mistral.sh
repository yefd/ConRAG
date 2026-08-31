


model_type="thot"
date="20250101"
prompt_type="qa_thot"

dataset_name="musique"

nohup python runner.py \
    --dataset_name $dataset_name \
    --train_data_path dataset/musique/musique_ans_v1.0_train_bert.pkl \
    --test_data_path dataset/musique/musique_ans_v1.0_dev_bert.pkl \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --freeze_llm \
    --prompt_type $prompt_type \
    --num_k 20 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/baselines/$model_type/output_${date}_${dataset_name}_mistral.log 2>&1 ;

nohup python runner.py \
    --dataset_name $dataset_name \
    --train_data_path dataset/musique/musique_ans_v1.0_train_bert.pkl \
    --test_data_path dataset/musique/musique_ans_v1.0_dev_bert.pkl \
    --model_name models/Llama-2-7b-chat-hf \
    --max_prompt_length 4096 \
    --freeze_llm \
    --prompt_type $prompt_type \
    --num_k 20 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/baselines/$model_type/output_${date}_${dataset_name}_llama2.log 2>&1 ;