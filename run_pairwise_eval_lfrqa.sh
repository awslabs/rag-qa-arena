##### This script run RAG-QA Arena by compare with LFRQA directly ######

# A few examples of model options,
# model1=meta-llama/Meta-Llama-3-70B-Instruct
# model1=mistralai/Mixtral-8x22B-Instruct-v0.1
# model1=CohereForAI/c4ai-command-r-plus
# model1=Qwen/Qwen1.5-32B-Chat
# model1=gpt-4-turbo
model1=gpt-4o

openai_key=''

eval_model=gpt-4-0125-preview
n_passages=5
domains=(bioasq fiqa recreation technology science writing lifestyle)
for i in "${!domains[@]}"
do
    echo evaluating ${model1} against LFRQA using ${eval_model} for ${domains[i]}
    python code/evaluate_pair_responses.py \
        --eval_dir "data/pairwise_eval" \
        --model ${eval_model} \
        --eval_model1 ${model1} \
        --model1_pred_file  ${model1}/${domains[i]}_from_colbert_${n_passages}_psgs  \
        --reference_file references \
        --template_config pairwise_lfrqa.cfg \
        --domain ${domains[i]} \
        --temperature 0.0 \
        --eval_input_save_dir eval_inputs/from_colbert \
        --api_key ${openai_key}
done