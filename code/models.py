from openai import OpenAI
import torch
from tqdm import tqdm
from utils import chunk, extract_output, extract_prompt, is_hf_model
import time

from transformers import (
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)

class GenericModel():
    def __init__(self, config):
        self.config = config

    def generate(self, data):
        pass

class HFModel(GenericModel):
    def __init__(self, config):
        super().__init__(config)

        if config.load_in_8bit and config.load_in_4bit:
            raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
        elif config.load_in_8bit or config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=config.load_in_8bit, 
                load_in_4bit=config.load_in_4bit
            )
            # This means: fit the entire model on a GPU
            device_map = "auto"
            torch_dtype = torch.float16
        else:
            device_map = "auto"
            quantization_config = None
            if config.bf16:
                torch_dtype = torch.bfloat16
            elif config.fp16:
                torch_dtype = torch.float16
            else:
                torch_dtype = None

        self.model = AutoModelForCausalLM.from_pretrained(
                    config.model,
                    device_map=device_map,
                    quantization_config=quantization_config,
                    torch_dtype=torch_dtype,
                    trust_remote_code=config.trust_remote_code,
                    use_auth_token=config.use_auth_token
            )

    def run_predictions(self, data, config, tokenizer=None):
        res = []
        pad_token_id = tokenizer.eos_token_id    
        for batch_example in tqdm(chunk(data, config.inference_batch_size)):
            input_ids = torch.tensor(batch_example["input_ids"])
            attention_mask = torch.tensor(batch_example["attention_mask"])
            preds = self.model.generate(input_ids=input_ids.to(self.model.device), 
                                        attention_mask=attention_mask.to(self.model.device), 
                                        max_new_tokens=self.config.max_new_tokens,
                                        do_sample=self.config.do_sample,
                                        pad_token_id=pad_token_id)
            # do not skip special tokens as they are useful to split prompt and pred
            decoded_preds = tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=False)
            for pred, example in zip(decoded_preds, batch_example):
                if tokenizer.eos_token and tokenizer.bos_token:
                    pred = pred.replace(tokenizer.eos_token, '').replace(tokenizer.bos_token, '')
                example['prompt'] = extract_prompt(pred, self.config.clm_new_token_trigger)
                example['pred'] = extract_output(pred, self.config.clm_new_token_trigger)
                # remove special tokens after split prompt and pred
                if tokenizer.additional_special_tokens:
                    for special_token in tokenizer.additional_special_tokens:
                        example['pred'] = example['pred'].replace(special_token, '')
                example.pop('input_ids')
                example.pop('attention_mask')
                res.append(example)
        return res


class OpenAIModel(GenericModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = OpenAI(api_key=config.api_key)
        self.model_name = config.model

    def run_predictions(self, data, args, generation_kwargs={}, tokenizer=None):
        res = []
        fail_count = 0
        for ex in tqdm(data):
            retry = 0
            # if pairwise comparison
            messages = []
            if 'ans_generation' not in args.template_config:
                messages.append({"role": "system", "content": ex["system"]})
                for turn in ex['examples']:
                    messages.append({"role": "user", "content": turn['user']})
                    messages.append({"role": "assistant", "content": turn['assistant']})
            messages.append({"role": "user", "content": ex['prompt']})
            while retry < 5:
                try:
                    ans = self.model.chat.completions.create(model=self.model_name, 
                                                             messages=messages,
                                                             max_tokens=args.max_new_tokens,
                                                             temperature=args.temperature,
                                                             top_p=args.top_p)
                    ex['pred'] = ans.choices[0].message.content
                    break
                except:
                    retry += 1
                    time.sleep(1)
            if retry == 5:
                fail_count += 1
                ex['pred'] = 'FAIL TO GENERATE ANS.'
                print(f"failed {fail_count} times")
            res.append(ex)
        
        return res


def load_model(args, logger):
    if 'gpt' in args.model:
        model = OpenAIModel(args)
    elif is_hf_model(args.model):
        model = HFModel(args)
    else:
        logger.info(f"{args.model} not supported!")
        exit()
    return model