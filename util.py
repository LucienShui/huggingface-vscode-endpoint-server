import logging
from typing import Tuple

import torch
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from transformers import GPT2TokenizerFast, GPTBigCodeForCausalLM, GPTBigCodeConfig

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('app')


def from_pretrained(pretrained: str, distributed: bool = False, max_memory: dict = None,
                    half: bool = True) -> Tuple[GPTBigCodeForCausalLM, GPT2TokenizerFast]:
    if distributed:
        config = GPTBigCodeConfig.from_pretrained(pretrained)
        with init_empty_weights():
            model: GPTBigCodeForCausalLM = GPTBigCodeForCausalLM.from_config(config)
        device_map = infer_auto_device_map(model, max_memory=max_memory)
        logger.info(f'device_map = {device_map}')
        model.tie_weights()
        model: GPTBigCodeForCausalLM = load_checkpoint_and_dispatch(
            model, pretrained, device_map=device_map, dtype=torch.float16)
    else:
        model: GPTBigCodeForCausalLM = GPTBigCodeForCausalLM.from_pretrained(pretrained)
        model: GPTBigCodeForCausalLM = model.half().cuda() if half else model.cuda()
    tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained(pretrained)
    return model, tokenizer
