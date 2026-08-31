

nohup python runner.py \
    --dataset_name bioasq \
    --train_data_path dataset/bioasq/bioasq_train.pkl \
    --test_data_path dataset/bioasq/bioasq_test.pkl \
    --model_name models/BioMistral-7B \
    --max_prompt_length 8192 \
    --freeze_llm \
    --num_k 10 \
    --model_type rag \
    --use_evaluation \
    --save_results \
    --instruction_type chat \
    >  logs/bioasq/output_rag_biomistral.log 2>&1 &