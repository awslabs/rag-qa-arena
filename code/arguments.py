from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict

@dataclass
class GlobalArguments:
    """
    Arguments global to a task or an evaluation run.
    """
    model: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    adapter_names: str = field(
        default=None,
        metadata={"help": "a list of adapters separated by ',' "}
    )
    adapters: str = field(
        default=None,
        metadata={"help": "a list of adapters path separated by ',' "}
    )
    combine_method: str = field(
        default='svd',
        metadata={"help": "linear, svd or cat "}
    )
    input_file: str = field(
        default=None,
        metadata={"help": "data to run answer generation with"}
    )
    output_file: str = field(
        default=None,
        metadata={"help": "file to save generation"}
    )
    n_passages: int = field(
        default=5,
        metadata={"help": "number of retrieved passages used as context"}
    )
    max_new_tokens: int = field(
        default=256,
        metadata={"help": "number of max generated tokens"}
    )
    num_beams: int = field(
        default=1,
        metadata={
            "help": (
                "Generation config. Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    do_sample: bool = field(
        default=False,
        metadata={
            "help": "Generation config. Whether to use sampling during generation."
        }
    )
    temperature: float = field(
        default=0.0,
        metadata={
            "help": "Generation config. What temperature to use during generation."
        }
    )
    top_k: int = field(
        default=None,
        metadata={
            "help": "Generation config. Top k vocab tokens to use during generation."
        }
    )
    top_p: float = field(
        default=1.0,
        metadata={
            "help": "Generation config. Top p-probability mass vocab tokens to use during generation."
        }
    )
    repetition_penalty: float = field(
        default=None,
        metadata={
            "help": "Generation config. The parameter for repetition penalty. 1.0 means no penalty."
        }
    )
    length_penalty: float = field(
        default=None,
        metadata={
            "help": "Generation config. Exponential penalty to the length. 1.0 means that the beam score is penalized by the sequence length. 0.0 means no penalty."
        }
    )
    diversity_penalty: float = field(
        default=None,
        metadata={
            "help": "Generation config. This value is subtracted from a beam’s score if it generates a token same as any beam from other group at a particular time."
        }
    )
    template_config: str = field(
        default=None,
        metadata={"help": "prompt template config file"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    bf16: Optional[bool] = field(default=False, metadata={"help": "whether to bf16"})
    fp16: Optional[bool] = field(default=False, metadata={"help": "whether to float16"})
    use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})
    device: Optional[int] = field(default=0, metadata={"help": "which gpu/cpu to use"})
    inference_batch_size: Optional[int] = field(default=1, metadata={"help": "batch size for decoding only"})
    reward_scale: Optional[int] = field(
        default=0, 
        metadata={"help": "auxilliary generation args for controllable generation"}
    )
    clm_new_token_trigger: Optional[str] = field(
        default=None,
        metadata={"help": "trigger word for CLM generations"}
    )
    max_sample: int = field(
        default=-1,
        metadata={"help": "max eval samples; default to -1: eval all"}
    )
    max_input_length: int = field(
        default=4096,
        metadata={"help": "max input token length; default to 4096"}
    )
    mode: str = field(
        default="generation",
        metadata={"help": "running answer generation or pairwise evaluation"}
    )
    local_rank: Optional[int] = field(default=-1, metadata={"help": "local rank"})
    deepspeed: Optional[str] = field(default=None, metadata={"help": "path to deepspeed config"})
    api_key: Optional[str] = field(default=None, metadata={"help": "OpenAI access token"})
    eval_dir: Optional[str] = field(default=None, metadata={"help": "ans gen output dir / evaluation input dir"})

@dataclass
class EvalArguments:
    """
    Arguments related to pairwise evaluation.
    """
    eval_model1: str = field(
        default=None,
        metadata={"help": "model 1 to be compared"}
    )
    eval_model2: str = field(
        default=None,
        metadata={"help": "model 2 to be compared"}
    )
    model1_pred_file: str = field(
        default=None,
        metadata={"help": "model 1 prediction file"}
    )
    model2_pred_file: str = field(
        default=None,
        metadata={"help": "model 2 prediction file"}
    )
    reference_file: str = field(
        default=None,
        metadata={"help": "reference file"}
    )
    aspect: str = field(
        default=None,
        metadata={"help": "aspect to evaluate: truthfulness, faithfulness, helpfulness"}
    )
    domain: str = field(
        default=None,
        metadata={"help": "data domain"}
    )
    eval_input_save_dir: str = field(
        default=None,
        metadata={"help": "intermediate dir to save processed evaluation files"}
    )
    eval_max_sample: int = field(
        default=-1,
        metadata={"help": "max eval samples; default to -1: eval all"}
    )
    seed: int = field(
        default=5,
        metadata={"help": "random seed to shuffle data"}
    )
    version: str = field(
        default="v0",
        metadata={"help": "eval template version"}
    )