import json
from transformers import (
    HfArgumentParser, 
    AutoTokenizer
)
from configobj import ConfigObj
import os
from models import load_model
from data_processors import DataProcessor
from arguments import GlobalArguments
from utils import get_logger, is_hf_model
logger = get_logger()

if __name__ == "__main__":
    parser = HfArgumentParser(GlobalArguments)
    parser.parse_args_into_dataclasses()
    args = parser.parse_args_into_dataclasses()[0]
    logger = get_logger(__name__)

    template_config = ConfigObj("templates" + os.path.sep + args.template_config)

    tokenizer = None
    if is_hf_model(args.model):
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    data_processor = DataProcessor(args, template_config['templateStr'], tokenizer)
    predict_dataset = data_processor.load_data()

    model = load_model(args, logger)
    data_to_save = model.run_predictions(predict_dataset, args, tokenizer=tokenizer)
    logger.info(f"Total {len(data_to_save)} predictions")
    
    # save prediction results
    output_dir = os.path.join(args.eval_dir, args.model)
    os.makedirs(output_dir, exist_ok=True)
    outfile = f'{args.output_file}_{args.n_passages}_psgs.json'
    with open(os.path.join(output_dir, outfile), 'w') as outfile:
        json.dump(data_to_save, outfile, indent=2)
        
