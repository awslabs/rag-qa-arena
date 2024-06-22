##### Script to run GPT-4 models #####

#model="gpt-4-turbo"
#model="gpt-4-0125-preview"
model="gpt-4o"

if [ ${model} == "gpt-4o" ]; then
    template_version=v2
else
    template_version=v1
fi

openai_key='' # you will need to enter your own OpenAI key here!

domains=(bioasq fiqa lifestyle recreation technology science writing)
for i in "${!domains[@]}"; do
    echo generating responses for ${domains[i]} using ${model} template ${template_version}
    python code/generate_responses.py \
    --model ${model} \
    --input_file data/ans_gen/${domains[i]}_from_colbert_test.jsonl \
    --output_file ${domains[i]}_from_colbert \
    --template_config ans_generation_${template_version}.cfg \
    --n_passages 5 \
    --eval_dir "data/pairwise_eval" \
    --api_key ${openai_key}
done