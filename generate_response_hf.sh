##### Script to run HF models #####

# WARNING: you will need to specify clm_new_token_trigger for different models!!
# Here is the mapping
# mistralai/Mixtral*:          "[/INST]"
# meta-llama/Meta-Llama*:      "<|start_header_id|>assistant<|end_header_id|>"
# CohereForAI/c4ai-command-r*: "<|CHATBOT_TOKEN|>"
# Qwen/Qwen1.5*:               "<|im_start|>assistant"


# model_path="meta-llama/Meta-Llama-3-70B-Instruct"
# model_path="CohereForAI/c4ai-command-r-v01"
# model_path="Qwen/Qwen1.5-32B-Chat"
# model_path="mistralai/Mixtral-8x7B-Instruct-v0.1"
model_path=mistralai/Mixtral-8x22B-Instruct-v0.1

domains=(bioasq fiqa lifestyle recreation technology science writing)
for i in "${!domains[@]}"; do
    devices="0,1,2,3,4,5,6,7"
    echo eval ${domains[i]} on ${devices} using ${model_path}
    export CUDA_VISIBLE_DEVICES=${devices}
    python code/generate_responses.py \
        --model ${model_path} \
        --input_file data/ans_gen/${domains[i]}_from_colbert_test.jsonl \
        --output_file ${domains[i]}_from_colbert \
        --template_config ans_generation_v1.cfg \
        --clm_new_token_trigger  "[/INST]" \
        --n_passages 5 \
        --eval_dir "data/pairwise_eval" \
        --fp16
done
