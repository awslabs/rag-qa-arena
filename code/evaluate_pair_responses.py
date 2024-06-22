import json
from transformers import (
    HfArgumentParser, 
    AutoTokenizer
)
from configobj import ConfigObj
import os
from models import load_model
from data_processors import PairDataProcessor, LFRQADataProcessor
from arguments import GlobalArguments, EvalArguments
from utils import get_logger, is_hf_model
logger = get_logger()

if __name__ == "__main__":
    parser = HfArgumentParser((GlobalArguments, EvalArguments))
    parser.parse_args_into_dataclasses()
    config, eval_args = parser.parse_args_into_dataclasses()
    logger = get_logger(__name__)

    template_config = ConfigObj("templates" + os.path.sep + config.template_config)

    tokenizer = None
    if is_hf_model(config.model):
        tokenizer = AutoTokenizer.from_pretrained(config.model)

    # if a pair of models' predictions are available
    if eval_args.eval_model1 and eval_args.eval_model2:
        data_processor = PairDataProcessor(config, template_config['templateStr'], tokenizer)
        predict_dataset = data_processor.load_data(eval_args)
        tempfile = f'{eval_args.domain}_{eval_args.eval_model1}_{eval_args.eval_model2}_eval_by_{config.model}.json'
    # using LFRQA as the target to compare, one LLM's prediction must be specified as eval_model1
    elif eval_args.eval_model1:
        data_processor = LFRQADataProcessor(config, template_config['templateStr'], tokenizer)
        predict_dataset = data_processor.load_data(eval_args)
        tempfile = f'{eval_args.domain}_{eval_args.eval_model1}_eval_by_{config.model}.json'
    else:
        logger.info(f"An LLM prediction file needs to be provided!")
    
    outpath = f'eval_results/lfrqa/{config.n_passages}_psgs/'
    os.makedirs(outpath, exist_ok=True)
    output_file_path = os.path.join(outpath, tempfile.replace('/', '-'))

    model = load_model(config, logger)
    data_to_save = model.run_predictions(predict_dataset, config, tokenizer=tokenizer)
    logger.info(f"Total {len(data_to_save)} predictions")
    
    with open(output_file_path, 'w') as outfile:
        json.dump(data_to_save, outfile, indent=2)
        
