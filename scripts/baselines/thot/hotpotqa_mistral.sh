


model_type="thot"
date="20250101"
prompt_type="qa_thot"

dataset_name="hotpotqa"

nohup python runner.py \
    --dataset_name $dataset_name \
    --train_data_path dataset/hotpotqa/hotpot_train_v1.1_bert.pkl \
    --test_data_path dataset/hotpotqa/hotpot_dev_distractor_v1_bert.pkl \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 4096 \
    --freeze_llm \
    --prompt_type $prompt_type \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/baselines/$model_type/output_${date}_${dataset_name}_rag_mistral.log 2>&1 ;

nohup python runner.py \
    --dataset_name $dataset_name \
    --train_data_path dataset/hotpotqa/hotpot_train_v1.1_bert.pkl \
    --test_data_path dataset/hotpotqa/hotpot_dev_distractor_v1_bert.pkl \
    --model_name models/Llama-2-7b-chat-hf \
    --max_prompt_length 4096 \
    --freeze_llm \
    --prompt_type $prompt_type \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/baselines/$model_type/output_${date}_${dataset_name}_rag_llama2.log 2>&1 ;