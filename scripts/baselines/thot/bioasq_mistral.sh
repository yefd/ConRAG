


model_type="thot"
date="20250101"
prompt_type="qa_thot"
dataset_name="bioasq"



nohup python runner.py \
    --dataset_name $dataset_name \
    --train_data_path dataset/bioasq/bioasq_train.pkl \
    --test_data_path dataset/bioasq/bioasq_test.pkl \
    --model_name models/Mistral-7B-Instruct-v0.3 \
    --max_prompt_length 8192 \
    --freeze_llm \
    --prompt_type $prompt_type \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/baselines/$model_type/output_${date}_${dataset_name}_mistral.log 2>&1 &