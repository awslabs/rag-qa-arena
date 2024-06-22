from template import TemplatedExample
import json 
from datasets import load_dataset
import os 
from utils import process_response, is_hf_model
import random

def _get_context_class(header_format="", idx_format="", title_format="", newline_format="\n</passage>\n"):
    template_str = header_format
    if idx_format:
        template_str += idx_format
    if title_format:
        template_str += title_format
    template_str += "{x.passage}" + newline_format
    class TemplatedContext(TemplatedExample):
        def __init__(self, passage, idx="", title=""):
            super().__init__(template_str)
            self.passage = passage
            if title_format:
                self.title = title
            if idx_format:
                 self.idx = idx

    return TemplatedContext

class ExampleTemplate(TemplatedExample):
    def __init__(self, passages, question, template_str):
        super().__init__(template_str)
        self.passages = passages
        self.question = question

class PairwiseEvalTemplate(TemplatedExample):
    def __init__(self, question, response1, response2, template_str, passages=[], reference=""):
        super().__init__(template_str)
        self.question = question
        self.response1 = response1
        self.response2 = response2
        if passages:
            self.passages = passages
        if reference:
            self.reference = reference

class DataProcessor():
    def __init__(self, config, template, tokenizer=None):
        self.config = config
        self.context_formatter = _get_context_class(header_format="<passage", idx_format="{x.idx}>\n")
        self.template_str = template
        self.tokenizer = tokenizer
        self.input_file = config.input_file

    def render_prompt_template(self, query, passages):
        templated_passages = [self.context_formatter(passage, i + 1) for i, passage in enumerate(passages)]

        example = ExampleTemplate(
            passages = templated_passages,
            question = query,
            template_str=self.template_str
        ).render()[0]
        
        return example

    def process_raw_data(self, args=None):
        processed_data = []
        with open(self.input_file) as infile:
            for line in infile:
                ex = json.loads(line)
                query = ex['query']
                passages = [psg['text'] for psg in ex['passages'][:self.config.n_passages]]
                example = self.render_prompt_template(query, passages)

                processed_data.append({'query': query, 'passages': passages, 'prompt': example})
                if self.config.max_sample > 0 and len(processed_data) >= self.config.max_sample:
                    break
                
        save_file = self.input_file.replace('.jsonl', '_processed_.jsonl')
        with open(save_file, 'w') as outfile:
            json.dump(processed_data, outfile, indent=2)

        return processed_data, save_file

    def hf_preprocess_function(self, examples):    
        batch_size = len(examples['query'])
        max_len = 0
        examples["full_prompt"] = examples["prompt"].copy()
        if self.config.mode == 'evaluation':
            for i, (inst, ex, prompt) in enumerate(zip(examples['system'], examples['examples'], examples['prompt'])):
                messages = []
                if 'mistral' in self.config.model:
                    messages.append({"role": "user", "content": inst})
                    messages.append({"role":"assistant", "content": "I understand."})
                else:
                    messages.append({"role": "system", "content": inst})
                for turn in ex:
                    messages.append({"role": "user", "content": turn['user']})
                    messages.append({"role": "assistant", "content": turn['assistant']})
                messages.append({"role": "user", "content": prompt})
                examples["full_prompt"][i] = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            for i, prompt in enumerate(examples['prompt']):
                messages = [{"role": "user", "content": prompt}]
                examples["full_prompt"][i] = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        model_inputs = self.tokenizer(examples['full_prompt'])
        max_len = 0
        for i in range(batch_size):
            L = len(model_inputs["input_ids"][i])
            max_len = max(L, max_len)
            if L > self.config.max_input_length:
                sample_input_ids = model_inputs["input_ids"][i][L-self.config.max_input_length:]
            else:
                sample_input_ids = model_inputs["input_ids"][i]
            model_inputs["input_ids"][i] = sample_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

        return model_inputs 
    
    def load_data(self, args=None):
        '''load data; if hf models; load them into HF datasets'''
        examples, save_file = self.process_raw_data(args)
        if is_hf_model(self.config.model):
            dataset = load_dataset('json', data_files={'test': save_file})['test']
            examples = dataset.map(
                                    self.hf_preprocess_function,
                                    batched=True,
                                    load_from_cache_file=False,
                                    desc="Running tokenizer on dataset",
                                )
        return examples


class LFRQADataProcessor(DataProcessor):
    def __init__(self, config, template, tokenizer=None):
        super().__init__(config, template, tokenizer)

        self.examples = self.render_examples()
        self.instruction = self.render_instruction()
        self.eval_dir = config.eval_dir
        self.ref_dir = "data"

    def render_examples(self):
        with open("templates/pairwise_lfrqa_examples.json") as infile:
            return json.load(infile)

    def render_instruction(self):
        instruction = ""
        with open("templates/pairwise_lfrqa_system.txt") as infile:
            for line in infile.readlines():
                instruction += line
        return instruction

    def render_prompt_template(self, query, response1, response2):

        example = PairwiseEvalTemplate(
            question=query,
            response1=response1,
            response2=response2,
            template_str=self.template_str
        ).render()[0]

        return example
    
    def create_data(self, infile_path, reference_path):
        '''load model predictions + filter w/o annotated references'''
        with open(reference_path) as infile:
            references = json.load(infile)

        q_predictions = {}
        no_reference_count, no_pred_count = 0, 0
        with open(infile_path) as infile:
            data = json.load(infile)
            for ex in data:
                query = ex['query']
                if query in references:
                    if 'pred' not in ex:
                        no_pred_count+=1
                    else:
                        q_predictions[query] = {'pred': ex['pred'],
                                                'reference': references[query]['faithful_answer']}
                else:
                    no_reference_count += 1
                    
        return q_predictions

    def process_raw_data(self, args):
        '''process and save data for hf loader later'''


        icl_examples = [{'user': self.render_prompt_template(ex['query'], ex['response_1'], ex['response_2']),
                        'assistant': f"<thinking>{ex['thinking']}</thinking><rating>{ex['label']}</rating>"} 
                        for ex in self.examples]
        
        processed_data = []
        # comparing model response with LFRQA
        reference_file = os.path.join(self.ref_dir, f"{args.domain}_{args.reference_file}.json")
        model_pred_file = os.path.join(self.eval_dir, f"{args.model1_pred_file}.json")
        preds = self.create_data(model_pred_file, reference_file)
    
        sample_count = 0
        for query, v in preds.items():
            pred = process_response(v['pred'])
            reference = process_response(v['reference'])
            if len(query.split(' ')) % 2 == 0:
                response1, response2 = pred, reference
                order = {1: args.model1_pred_file.split('/')[0], 2: 'LFRQA'}
            else:
                response1, response2 = reference, pred
                order = {1: 'LFRQA', 2: args.model1_pred_file.split('/')[0]}

            example = self.render_prompt_template(query, response1, response2)
            processed_data.append({'pair_id': sample_count, 
                                   'system': self.instruction,
                                   'examples': icl_examples,
                                   'prompt': example, 
                                   'query': query,
                                   'response_1': response1, 
                                   'response_2': response2, 
                                   'order': order})

            if self.config.max_sample > 0 and len(processed_data) >= self.config.max_sample:
                break
            sample_count += 1
        save_file = f"{args.domain}_{args.eval_model1}.json"

        if not os.path.exists(args.eval_input_save_dir):
            os.makedirs(args.eval_input_save_dir)
            
        save_file = os.path.join(args.eval_input_save_dir, save_file.replace('/', '-'))
        with open(save_file, 'w') as outfile:
            if args.eval_max_sample > 0:
                random.Random(args.seed).shuffle(processed_data)
                processed_data = processed_data[:args.eval_max_sample]
            json.dump(processed_data, outfile, indent=2)

        return processed_data, save_file
    
class PairDataProcessor(LFRQADataProcessor):
    
    def merge_data(self, infile_path, reference_path):
        '''load model predictions + filter w/o annotated references'''
        with open(reference_path) as infile:
            references = json.load(infile)

        q_passage_predictions = {}
        no_reference_count, no_pred_count = 0, 0
        with open(infile_path) as infile:
            data = json.load(infile)
            for ex in data:
                query = ex['query']
                if query in references:
                    if 'pred' not in ex:
                        no_pred_count+=1
                    else:
                        q_passage_predictions[query] = {'passages': ex['passages'], 
                                                        'pred': ex['pred'],
                                                        'reference': references[query]['faithful_answer']}
                else:
                    no_reference_count += 1
                    
        return q_passage_predictions

    def process_raw_data(self, args):
        '''process and save data for hf loader later'''
        reference_file = os.path.join(self.ref_dir, f"{args.domain}_{args.reference_file}.json")
        model1_pred_file = os.path.join(self.eval_dir, f"{args.model1_pred_file}.json")
        model2_pred_file = os.path.join(self.eval_dir, f"{args.model2_pred_file}.json")
        model1_preds = self.merge_data(model1_pred_file, reference_file)
        model2_preds = self.merge_data(model2_pred_file, reference_file)

        icl_examples = [{'user': self.render_prompt_template(ex['query'], ex['response_1'], ex['response_2']),
                        'assistant': f"<thinking>{ex['thinking']}</thinking><rating>{ex['label']}</rating>"} 
                        for ex in self.examples]
        
        processed_data = []
        sample_count = 0
        for query, v in model1_preds.items():
            pred1 = process_response(v['pred'])
            if query in model2_preds:
                pred2 = process_response(model2_preds[query]['pred'])
                if len(query.split(' ')) % 2 == 0:
                    response1, response2 = pred1, pred2
                    order = {1: args.eval_model1, 2: args.eval_model2}
                else:
                    response1, response2 = pred2, pred1
                    order = {1: args.eval_model2, 2: args.eval_model1}

                example = self.render_prompt_template(query, response1, response2)
                processed_data.append({'pair_id': sample_count,
                                    'system': self.instruction,
                                    'examples': icl_examples,
                                    'prompt': example, 
                                    'query': query, 'passages': v['passages'], 'reference': v['reference'],
                                    'response_1': response1, 'response_2': response2, 'order': order})
                if self.config.max_sample > 0 and len(processed_data) >= self.config.max_sample:
                    break
                sample_count += 1

        if not os.path.exists(args.eval_input_save_dir):
            os.makedirs(args.eval_input_save_dir)
            
        save_file = f"{args.domain}_{args.eval_model1}_{args.eval_model2}.json"
        save_file = os.path.join(args.eval_input_save_dir, save_file.replace('/', '-'))
        with open(save_file, 'w') as outfile:
            if args.eval_max_sample > 0:
                random.Random(args.seed).shuffle(processed_data)
                processed_data = processed_data[:args.eval_max_sample]
            json.dump(processed_data, outfile, indent=2)

        return processed_data, save_file